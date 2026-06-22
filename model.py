"""
Pipeline-parallel GPT stage 模型（纯 PyTorch，无外部依赖）。

Ring 拓扑布局（LM head 集中到 stage 0）：
  Stage 0      : Embedding + L 层 Transformer + final LN + LM Head
                 forward 拆成两个具名方法对应 ring 上两个物理时机：
                   · forward_embed(input_ids) → hidden     (mb 起始)
                   · forward_head(hidden)     → logits      (mb 末尾，从 rank K-1 收回)
  Stage 1..K-2 : 中间 L 层 Transformer
  Stage K-1    : 纯 L 层 Transformer（无 ln_f，无 lm_head）—— forward 完把 hidden 发回 rank 0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, H, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = H // num_heads
        self.qkv = nn.Linear(H, 3 * H)
        self.out = nn.Linear(H, H)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self.out(y.transpose(1, 2).contiguous().view(B, T, C))


class TransformerBlock(nn.Module):
    def __init__(self, H, num_heads, dropout):
        super().__init__()
        self.ln1  = nn.LayerNorm(H)
        self.attn = CausalSelfAttention(H, num_heads, dropout)
        self.ln2  = nn.LayerNorm(H)
        self.mlp  = nn.Sequential(
            nn.Linear(H, 4 * H), nn.GELU(),
            nn.Linear(4 * H, H), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class StageFirst(nn.Module):
    """Stage 0: 双重职责。

    forward_embed: input_ids → hidden_h0    （mb 起始，发给 rank 1）
    forward_head : hidden_hK → logits        （mb 末尾，从 rank K-1 收回；后续接 cross_entropy）

    注意：__call__ / forward 不暴露，调用方必须显式选择两个方法之一。
    """
    def __init__(self, cfg):
        super().__init__()
        H = cfg["hidden_dim"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], H)
        self.pos_emb = nn.Embedding(cfg["max_seq_len"], H)
        self.drop    = nn.Dropout(cfg["dropout"])
        self.blocks  = nn.ModuleList(
            [TransformerBlock(H, cfg["num_heads"], cfg["dropout"])
             for _ in range(cfg["num_layers_per_stage"])]
        )
        self.ln_f    = nn.LayerNorm(H)
        self.lm_head = nn.Linear(H, cfg["vocab_size"], bias=False)

    def forward_embed(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward_head(self, hidden):
        return self.lm_head(self.ln_f(hidden))


class StageMiddle(nn.Module):
    """中间 stage: hidden_states → hidden_states"""
    def __init__(self, cfg):
        super().__init__()
        H = cfg["hidden_dim"]
        self.blocks = nn.ModuleList(
            [TransformerBlock(H, cfg["num_heads"], cfg["dropout"])
             for _ in range(cfg["num_layers_per_stage"])]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class StageLast(nn.Module):
    """Stage K-1: 纯 L 层 Transformer。

    Ring 拓扑下结构等同 StageMiddle —— forward 完把 hidden 通过新通道发回 rank 0。
    保留独立类是为 build_stage() 的角色身份清晰；不复制为 StageMiddle 别名。
    """
    def __init__(self, cfg):
        super().__init__()
        H = cfg["hidden_dim"]
        self.blocks = nn.ModuleList(
            [TransformerBlock(H, cfg["num_heads"], cfg["dropout"])
             for _ in range(cfg["num_layers_per_stage"])]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


def build_stage(rank: int, num_stages: int, cfg: dict) -> nn.Module:
    if rank == 0:
        return StageFirst(cfg)
    elif rank == num_stages - 1:
        return StageLast(cfg)
    else:
        return StageMiddle(cfg)
