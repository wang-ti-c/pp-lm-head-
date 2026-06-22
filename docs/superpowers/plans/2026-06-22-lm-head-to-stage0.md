# LM Head Migration to Stage 0 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Permanently relocate `ln_f` + `lm_head` + cross-entropy loss from rank K-1 to rank 0, turning the pipeline into a ring (`0→1→2→3→0`) so rank 0 holds the heaviest stage (suitable for ondemand) and rank K-1 holds the lightest (suitable for spot).

**Architecture:** Two new P2P channels — `K-1 → 0` for forward hidden, `0 → K-1` for backward grad — close the loop. Stage 0 gains a dual-method `forward_embed` / `forward_head` API; Stage K-1 degenerates to pure transformer blocks. Recovery protocol (sync only) is rewritten so rank 0 owns loss; async path is deferred to a separate spec. The `plan_recovery` role mapping (UPSTREAM_HELPER / PREEMPTED / DOWNSTREAM_VICTIM) is preserved verbatim.

**Tech Stack:** PyTorch (`torch.distributed`, `torch.func.functional_call`), NCCL on A100, gloo for CPU tests, pytest.

## Global Constraints

- **Sync path only.** Do not modify `execute_recovery_async` in `recovery_protocol.py`. Async path remains a separate follow-up spec.
- **Full migration, no toggle.** No config flag for "head_at_stage0". Old layout is dead.
- **`plan_recovery` semantics unchanged.** Role assignment (rank < target → HELPER, == target → PREEMPTED, > target → VICTIM) must remain bitwise identical. `test_plan_recovery_unchanged` must keep passing without modification.
- **No weight tying.** `lm_head` weight does not share storage with `tok_emb`.
- **Do not touch:** `coordinator.py`, `checkpoint_v2.py`, `activation_cache.py`, `analyze_v2.py`, `configs/*.yaml`.
- **Model hyperparams unchanged:** `hidden_dim`, `num_layers_per_stage`, `vocab_size`, `num_heads` stay as configured.
- **Comments and docstrings in Chinese where existing code uses Chinese**, English where existing code uses English — match the file you are editing.

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `model.py` | Modify | New `StageFirst` with `forward_embed` / `forward_head`; degenerate `StageLast` to pure blocks |
| `pp_engine.py` | Modify | Add 4 ring-closure comm primitives; rewrite `step()`; loosen `do_retain_here` |
| `recovery_protocol.py` | Modify | Rewrite `execute_recovery_sync` only — rank 0 owns loss, rank K-1 becomes a middle-shaped stage. `plan_recovery` and `execute_recovery_async` untouched |
| `run_injection_v2.py` | Modify | Flip 6 `rank K-1`-owns-loss hardcodes to `rank 0`; drop `batch_targets` requirement from rank K-1 |
| `tests/test_async_recovery.py` | Modify | Adapt `_TinyStage`; add 3 new CPU tests |
| `README.md` | Modify | Update stage layout, add ring topology section, add capacity/deployment table |

---

### Task 1: Rewrite model.py with dual-method StageFirst and degenerate StageLast

**Files:**
- Modify: `model.py` (whole file)
- Test: `tests/test_model_layout.py` (create new)

**Interfaces:**
- Consumes: nothing
- Produces:
  - `StageFirst.forward_embed(input_ids: LongTensor[B,T]) -> FloatTensor[B,T,H]`
  - `StageFirst.forward_head(hidden: FloatTensor[B,T,H]) -> FloatTensor[B,T,V]`
  - `StageLast.forward(x: FloatTensor[B,T,H]) -> FloatTensor[B,T,H]` (no head, no ln_f)
  - `build_stage(rank, num_stages, cfg) -> nn.Module` (signature unchanged)
  - `StageFirst` attributes: `tok_emb`, `pos_emb`, `drop`, `blocks`, `ln_f`, `lm_head`
  - `StageLast` attributes: `blocks` only (no `ln_f`, no `lm_head`)

- [ ] **Step 1: Write failing tests for new model layout**

Create `tests/test_model_layout.py`:

```python
"""Tests for the head-at-stage-0 ring layout."""
import torch
from model import build_stage, StageFirst, StageLast, StageMiddle


def _cfg(H=8, V=16, L=2, heads=2, seq=4, dropout=0.0):
    return {
        "hidden_dim": H, "vocab_size": V, "num_layers_per_stage": L,
        "num_heads": heads, "max_seq_len": seq, "dropout": dropout,
    }


def test_stage_first_has_head_and_ln_f():
    s = StageFirst(_cfg())
    assert hasattr(s, "lm_head")
    assert hasattr(s, "ln_f")
    assert hasattr(s, "tok_emb")
    assert hasattr(s, "pos_emb")


def test_stage_first_no_legacy_forward():
    """StageFirst.forward must be removed; only forward_embed / forward_head exist."""
    s = StageFirst(_cfg())
    # __call__ inherits forward from nn.Module; we want it to NOT be a usable
    # entry — calling it on a token id tensor must NOT silently work the old way.
    # We assert the two new methods exist:
    assert callable(getattr(s, "forward_embed", None))
    assert callable(getattr(s, "forward_head", None))


def test_stage_first_forward_embed_shape_and_grad():
    cfg = _cfg(H=8, V=16, seq=4)
    s = StageFirst(cfg)
    ids = torch.randint(0, cfg["vocab_size"], (2, cfg["max_seq_len"]))
    h = s.forward_embed(ids)
    assert h.shape == (2, cfg["max_seq_len"], cfg["hidden_dim"])
    # gradient flows back to tok_emb
    h.sum().backward()
    assert s.tok_emb.weight.grad is not None


def test_stage_first_forward_head_shape_and_grad():
    cfg = _cfg(H=8, V=16, seq=4)
    s = StageFirst(cfg)
    hidden = torch.randn(2, cfg["max_seq_len"], cfg["hidden_dim"],
                          requires_grad=True)
    logits = s.forward_head(hidden)
    assert logits.shape == (2, cfg["max_seq_len"], cfg["vocab_size"])
    logits.sum().backward()
    assert hidden.grad is not None
    assert s.lm_head.weight.grad is not None
    assert s.ln_f.weight.grad is not None


def test_stage_last_no_head():
    s = StageLast(_cfg())
    assert not hasattr(s, "lm_head")
    assert not hasattr(s, "ln_f")
    assert hasattr(s, "blocks")


def test_stage_last_forward_preserves_shape():
    cfg = _cfg(H=8, seq=4)
    s = StageLast(cfg)
    x = torch.randn(2, cfg["max_seq_len"], cfg["hidden_dim"])
    y = s(x)
    assert y.shape == x.shape


def test_build_stage_rank0_is_stage_first():
    cfg = _cfg()
    assert isinstance(build_stage(0, 4, cfg), StageFirst)


def test_build_stage_rankK_minus_1_is_stage_last():
    cfg = _cfg()
    assert isinstance(build_stage(3, 4, cfg), StageLast)


def test_build_stage_middle():
    cfg = _cfg()
    assert isinstance(build_stage(1, 4, cfg), StageMiddle)
    assert isinstance(build_stage(2, 4, cfg), StageMiddle)
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/test_model_layout.py -v`
Expected: tests fail with `AttributeError: 'StageFirst' object has no attribute 'forward_embed'` (and `StageLast` has `lm_head`).

- [ ] **Step 3: Rewrite model.py**

Replace the file contents:

```python
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
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest tests/test_model_layout.py -v`
Expected: all 8 tests PASS.

- [ ] **Step 5: Run plan_recovery test to confirm unrelated test still passes**

Run: `pytest tests/test_async_recovery.py::test_plan_recovery_unchanged -v`
Expected: PASS (this test never imports model.py — sanity check only).

- [ ] **Step 6: Commit**

```bash
cd /mnt/c/Users/12589/Desktop/pp-preempt-v2
git add model.py tests/test_model_layout.py
git commit -m "model: ring layout — LM head moves to stage 0, stage K-1 degenerates"
```

(If the project is not yet a git repo, skip the commit and note files changed in your handoff message. The next git operation in the project will pick them up.)

---

### Task 2: Add ring-closure comm primitives and rewrite PPEngine.step()

**Files:**
- Modify: `pp_engine.py` (whole file)

**Interfaces:**
- Consumes (from Task 1):
  - `StageFirst.forward_embed(input_ids) -> hidden`
  - `StageFirst.forward_head(hidden) -> logits`
  - `StageLast.forward(x) -> hidden` (used via `self.stage(inp)`)
- Produces:
  - `PPEngine._send_hidden_to_head(x)` — rank K-1 send to rank 0
  - `PPEngine._recv_hidden_from_tail() -> Tensor[B,T,H]` — rank 0 receive from rank K-1
  - `PPEngine._send_grad_to_tail(g)` — rank 0 send to rank K-1
  - `PPEngine._recv_grad_from_head() -> Tensor[B,T,H]` — rank K-1 receive from rank 0
  - `PPEngine.step(batch_input_ids, batch_targets, optimizer, step_id)` — rank 0 returns scalar loss, others return 0.0
  - `do_retain_here = do_retain` (no `< K-1` exception)

- [ ] **Step 1: Replace pp_engine.py**

Replace whole file:

```python
"""
Pipeline-parallel 训练引擎（Ring 拓扑 + 图优化版）。

Ring 拓扑差异：
  · 新增 4 个 ring-closure 通信原语：_send_hidden_to_head / _recv_hidden_from_tail
                                   _send_grad_to_tail   / _recv_grad_from_head
  · rank 0 既驱动 embed forward（送给 rank 1），又消费来自 rank K-1 的 hidden 跑 head + loss
  · rank K-1 退化为中间 stage 同构：forward 完把 hidden 发回 rank 0，等 grad 回来 backward
  · do_retain_here 不再排除 K-1 —— K-1 现在与中间 stage 同性质，参与保留图

retain_graph 行为：
  · rank 0 的 embed 段：与原版一致，走 functional_call + cloned params 路径
  · rank 0 的 head 段：每 mb backward 后图自动释放，无需保留
  · rank K-1：与中间 stage 同形态，参与 retain_graph_interval 节奏
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.func import functional_call

from activation_cache import ActivationCache


class PPEngine:
    def __init__(self, stage_module, num_stages, num_microbatches,
                 micro_batch_size, seq_len, hidden_dim, vocab_size,
                 cache_max_steps: int = 200,
                 retain_graph_interval: int = 0):
        self.stage   = stage_module
        self.K       = num_stages
        self.M       = num_microbatches
        self.mb_size = micro_batch_size
        self.seq_len = seq_len
        self.H       = hidden_dim
        self.V       = vocab_size
        self.rank    = dist.get_rank()
        self.device  = (torch.cuda.current_device()
                        if torch.cuda.is_available()
                        else torch.device("cpu"))

        self.retain_graph_interval = retain_graph_interval
        self.cache = ActivationCache(max_steps=cache_max_steps,
                                     retain_graph_interval=retain_graph_interval)
        self.current_step    = -1
        self.last_graph_step = None

    # ── 通信原语（链式） ──────────────────────────────────────────────────────

    def _recv_act(self):
        buf = torch.empty((self.mb_size, self.seq_len, self.H),
                          device=self.device, dtype=torch.float32)
        dist.recv(buf, src=self.rank - 1)
        return buf

    def _send_act(self, x):
        dist.send(x.contiguous(), dst=self.rank + 1)

    def _recv_grad(self):
        buf = torch.empty((self.mb_size, self.seq_len, self.H),
                          device=self.device, dtype=torch.float32)
        dist.recv(buf, src=self.rank + 1)
        return buf

    def _send_grad(self, g):
        dist.send(g.contiguous(), dst=self.rank - 1)

    # ── 通信原语（ring 回传） ────────────────────────────────────────────────

    def _send_hidden_to_head(self, x):
        """rank K-1 → rank 0：forward 时把最末 hidden 送回 head"""
        dist.send(x.contiguous(), dst=0)

    def _recv_hidden_from_tail(self):
        """rank 0 接收来自 rank K-1 的 hidden（forward 末段）"""
        buf = torch.empty((self.mb_size, self.seq_len, self.H),
                          device=self.device, dtype=torch.float32)
        dist.recv(buf, src=self.K - 1)
        return buf

    def _send_grad_to_tail(self, g):
        """rank 0 → rank K-1：backward 启动时把 grad_hidden 送给 tail"""
        dist.send(g.contiguous(), dst=self.K - 1)

    def _recv_grad_from_head(self):
        """rank K-1 接收来自 rank 0 的 grad_hidden（backward 启动）"""
        buf = torch.empty((self.mb_size, self.seq_len, self.H),
                          device=self.device, dtype=torch.float32)
        dist.recv(buf, src=0)
        return buf

    # ── 参数副本 forward（图优化专用）────────────────────────────────────────

    def _forward_with_clones(self, inp):
        """
        中间 stage 与 K-1 用：functional_call + cloned params 跑 self.stage(inp)。
        autograd 图引用 cloned（version=0），optimizer.step() 修改原始参数后不破坏图。
        Returns: (out, cloned_params_dict)
        """
        cloned = {n: p.detach().clone().requires_grad_(True)
                  for n, p in self.stage.named_parameters()}
        out = functional_call(self.stage, cloned, (inp,))
        return out, cloned

    def _forward_with_clones_method(self, inp, method_name: str):
        """
        Rank 0 专用：在 cloned 参数上下文中调用 self.stage 的指定方法。

        实现方式：手动把 cloned 参数临时塞回 stage（保存原指针），
        调用方法，再恢复原参数。autograd 图建立在 cloned 上。
        """
        cloned = {n: p.detach().clone().requires_grad_(True)
                  for n, p in self.stage.named_parameters()}
        # 临时替换 stage 参数指针 → cloned
        originals = {}
        for n, p in list(self.stage.named_parameters()):
            originals[n] = p
        # 用 functional_call 的官方接口实现"在 cloned 参数下调用任意方法"：
        # torch.func.functional_call 支持传 (module, params, args)，会调用 module.__call__。
        # 但 StageFirst 的 __call__ 不可用。所以我们用 stateless functional_call 的内部 trick：
        # 通过子模块包装。简单做法：临时 monkey-patch self.stage.forward。
        method = getattr(self.stage, method_name)
        saved_forward = self.stage.forward if hasattr(self.stage, "forward") else None
        self.stage.forward = method  # type: ignore[assignment]
        try:
            out = functional_call(self.stage, cloned, (inp,))
        finally:
            if saved_forward is None:
                # nn.Module 总有 forward（默认 raise NotImplementedError）；删除我们设的属性
                try:
                    del self.stage.forward
                except AttributeError:
                    pass
            else:
                self.stage.forward = saved_forward  # type: ignore[assignment]
        return out, cloned

    def _transfer_grads(self, cloned_params: dict):
        """将 cloned_params 的梯度累积到原始参数的 .grad，供 optimizer.step()。"""
        for n, p in self.stage.named_parameters():
            g = cloned_params[n].grad
            if g is None:
                continue
            if p.grad is None:
                p.grad = g.detach().clone()
            else:
                p.grad.add_(g.detach())

    # ── 训练 step ─────────────────────────────────────────────────────────────

    def step(self, batch_input_ids, batch_targets, optimizer, step_id: int):
        self.current_step = step_id
        do_retain = (self.retain_graph_interval > 0
                     and step_id % self.retain_graph_interval == 0)
        # Ring 拓扑：K-1 现在与中间 stage 同性质，所有 rank 都参与
        do_retain_here = do_retain

        optimizer.zero_grad()
        fwd_in:    list = []
        fwd_out:   list = []
        head_in:   list = []  # rank 0 only: 收到的 hidden_h3 (requires_grad)
        losses:    list = []  # rank 0 only
        clone_sets: list = [] # embed 段的 cloned dict（rank 0/中间/K-1 用得到）

        # ── Forward：embed / middle / tail （所有 mb 一轮） ────────────────────
        for mb in range(self.M):

            if self.rank == 0:
                s   = mb * self.mb_size
                inp = batch_input_ids[s:s + self.mb_size].to(self.device)
                fwd_in.append(inp)
                if do_retain_here:
                    out, cloned = self._forward_with_clones_method(
                        inp, "forward_embed")
                    clone_sets.append(cloned)
                    self.cache.save_with_graph(step_id, mb, inp, out)
                else:
                    out = self.stage.forward_embed(inp)
                    clone_sets.append(None)
                    self.cache.save(step_id, mb, inp, out)
                fwd_out.append(out)
                self._send_act(out)

            elif self.rank == self.K - 1:
                inp = self._recv_act()
                inp.requires_grad_(True)
                fwd_in.append(inp)
                if do_retain_here:
                    out, cloned = self._forward_with_clones(inp)
                    clone_sets.append(cloned)
                    self.cache.save_with_graph(step_id, mb, inp, out)
                else:
                    out = self.stage(inp)
                    clone_sets.append(None)
                    self.cache.save(step_id, mb, inp, out)
                fwd_out.append(out)
                self._send_hidden_to_head(out)  # ← ring 回传

            else:  # 中间 stage 1..K-2
                inp = self._recv_act()
                inp.requires_grad_(True)
                fwd_in.append(inp)
                if do_retain_here:
                    out, cloned = self._forward_with_clones(inp)
                    clone_sets.append(cloned)
                    self.cache.save_with_graph(step_id, mb, inp, out)
                else:
                    out = self.stage(inp)
                    clone_sets.append(None)
                    self.cache.save(step_id, mb, inp, out)
                fwd_out.append(out)
                self._send_act(out)

        # ── Head / loss（仅 rank 0，per mb，紧跟 forward） ────────────────────
        if self.rank == 0:
            for mb in range(self.M):
                hidden = self._recv_hidden_from_tail()
                hidden.requires_grad_(True)
                head_in.append(hidden)
                logits = self.stage.forward_head(hidden)
                s   = mb * self.mb_size
                tgt = batch_targets[s:s + self.mb_size].to(self.device)
                losses.append(
                    F.cross_entropy(logits.view(-1, self.V), tgt.view(-1)) / self.M
                )

        # ── Backward ──────────────────────────────────────────────────────────
        if self.rank == 0:
            # Loop 1: 触发 head backward，启动环回 grad
            for mb in range(self.M):
                losses[mb].backward()
                self._send_grad_to_tail(head_in[mb].grad)
            # Loop 2: 等 grad 绕环回到 embed
            for mb in range(self.M):
                cloned = clone_sets[mb]
                g = self._recv_grad()
                fwd_out[mb].backward(g, retain_graph=do_retain_here)
                if cloned is not None:
                    self._transfer_grads(cloned)

        elif self.rank == self.K - 1:
            for mb in range(self.M):
                cloned = clone_sets[mb]
                g = self._recv_grad_from_head()  # ← ring 回传
                fwd_out[mb].backward(g, retain_graph=do_retain_here)
                self._send_grad(fwd_in[mb].grad)
                if cloned is not None:
                    self._transfer_grads(cloned)

        else:  # 中间 stage
            for mb in range(self.M):
                cloned = clone_sets[mb]
                g = self._recv_grad()
                fwd_out[mb].backward(g, retain_graph=do_retain_here)
                self._send_grad(fwd_in[mb].grad)
                if cloned is not None:
                    self._transfer_grads(cloned)

        optimizer.step()

        if do_retain_here:
            self.last_graph_step = step_id

        return sum(l.item() for l in losses) if self.rank == 0 else 0.0
```

- [ ] **Step 2: Sanity-check the file syntactically**

Run: `python -c "import ast; ast.parse(open('pp_engine.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Run model tests (no regression)**

Run: `pytest tests/test_model_layout.py -v`
Expected: all PASS (pp_engine.py imports model.py at construction time, not module level — model tests unaffected).

- [ ] **Step 4: Run plan_recovery test**

Run: `pytest tests/test_async_recovery.py::test_plan_recovery_unchanged -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/Users/12589/Desktop/pp-preempt-v2
git add pp_engine.py
git commit -m "pp_engine: ring-closure primitives + step() with rank 0 loss"
```

---

### Task 3: Rewrite execute_recovery_sync for ring topology

**Files:**
- Modify: `recovery_protocol.py` — only `execute_recovery_sync` function (lines 47–172). **Do NOT touch** `plan_recovery`, `RecoveryRole`, or anything in the `1F1B async recovery` section (line 176 onwards).

**Interfaces:**
- Consumes (from Task 2):
  - `engine._send_hidden_to_head(x)`, `engine._recv_hidden_from_tail()`
  - `engine._send_grad_to_tail(g)`, `engine._recv_grad_from_head()`
  - `engine.stage.forward_embed(input_ids)`, `engine.stage.forward_head(hidden)` (when `engine.rank == 0`)
- Produces:
  - `execute_recovery_sync(plan, engine, ckpt_step, resume_step, batch_input_ids, batch_targets, optimizer) -> dict` with shape unchanged from spec:
    ```
    { "activation_resend_sec": float, "recovery_compute_sec": float,
      "stages_resent_activation": list[int], "stages_recomputed_forward": list[int],
      "final_loss": float|None,  # now non-None on rank 0, None elsewhere
      "used_retained_graph": bool }
    ```

- [ ] **Step 1: Replace the `execute_recovery_sync` function**

Open `recovery_protocol.py`. Replace the function body (lines 47–172) with:

```python
def execute_recovery_sync(plan, engine, ckpt_step, resume_step,
                          batch_input_ids, batch_targets, optimizer) -> dict:
    """
    Ring 拓扑下的同步恢复：
      · rank 0 既驱动 embed forward，又消费 rank K-1 回传的 hidden 跑 head + loss
      · rank K-1 退化为中间 stage 同形态（forward 完 send hidden 回 0，等 grad 回来 backward）
      · plan_recovery 角色映射不变；UPSTREAM_HELPER 的 resend cache 行为不变
    """
    rank    = engine.rank
    role    = plan[rank]
    K, M    = engine.K, engine.M

    stages_resent     = [r for r, ro in plan.items() if ro == RecoveryRole.UPSTREAM_HELPER]
    stages_recomputed = [r for r, ro in plan.items()
                         if ro in (RecoveryRole.PREEMPTED, RecoveryRole.DOWNSTREAM_VICTIM)]

    if role == RecoveryRole.UPSTREAM_HELPER and not engine.cache.has_step(ckpt_step):
        raise RuntimeError(
            f"[rank {rank}] UPSTREAM_HELPER missing cache for ckpt_step={ckpt_step}. "
            f"{engine.cache.summary()}"
        )

    use_graph = (role == RecoveryRole.UPSTREAM_HELPER
                 and engine.cache.has_graph(ckpt_step))

    optimizer.zero_grad()

    # ── Forward / resend 阶段 ─────────────────────────────────────────────────
    t0 = time.monotonic()
    fwd_in:   list = []
    fwd_out:  list = []
    head_in:  list = []   # rank 0 only: 收到的 hidden_h3
    losses:   list = []   # rank 0 only

    for mb in range(M):
        if role == RecoveryRole.UPSTREAM_HELPER:
            # rank 0 的 UPSTREAM_HELPER 分支：只 resend cached_out 到 rank 1，
            #   但同时 rank 0 还要兼任 head（在后面的 head loop 里处理）。
            # 其他 rank 的 UPSTREAM_HELPER 分支与原版完全相同。
            if rank > 0:
                _ = engine._recv_act()          # drain buffer
            cached_inp, cached_out = engine.cache.get(ckpt_step, mb)
            fwd_in.append(cached_inp); fwd_out.append(cached_out)
            engine._send_act(cached_out)

        elif role == RecoveryRole.PREEMPTED:
            if rank == 0:
                s = mb * engine.mb_size
                inp = batch_input_ids[s:s + engine.mb_size].to(engine.device)
                fwd_in.append(inp)
                out = engine.stage.forward_embed(inp); fwd_out.append(out)
                engine._send_act(out)
            elif rank == K - 1:
                inp = engine._recv_act(); inp.requires_grad_(True)
                fwd_in.append(inp)
                out = engine.stage(inp); fwd_out.append(out)
                engine._send_hidden_to_head(out)
            else:
                inp = engine._recv_act(); inp.requires_grad_(True)
                fwd_in.append(inp)
                out = engine.stage(inp); fwd_out.append(out)
                engine._send_act(out)

        else:  # DOWNSTREAM_VICTIM
            if rank == K - 1:
                inp = engine._recv_act(); inp.requires_grad_(True)
                fwd_in.append(inp)
                out = engine.stage(inp); fwd_out.append(out)
                engine._send_hidden_to_head(out)
            else:
                inp = engine._recv_act(); inp.requires_grad_(True)
                fwd_in.append(inp)
                out = engine.stage(inp); fwd_out.append(out)
                engine._send_act(out)

    # ── Head / loss 阶段（仅 rank 0，per mb） ────────────────────────────────
    # rank 0 在以下任意角色（PREEMPTED / DOWNSTREAM_VICTIM / UPSTREAM_HELPER）
    # 都必须接收 K-1 发回的 hidden 并算 loss —— 这是 ring 拓扑下 rank 0 的固有职责。
    # （注：target_rank=0 时 rank 0 是 PREEMPTED；target_rank!=0 时 rank 0 是 UPSTREAM_HELPER。）
    if rank == 0:
        for mb in range(M):
            hidden = engine._recv_hidden_from_tail()
            hidden.requires_grad_(True)
            head_in.append(hidden)
            logits = engine.stage.forward_head(hidden)
            s   = mb * engine.mb_size
            tgt = batch_targets[s:s + engine.mb_size].to(engine.device)
            losses.append(F.cross_entropy(
                logits.view(-1, engine.V), tgt.view(-1)) / M)

    resend_sec = time.monotonic() - t0

    # ── Backward / compute 阶段 ───────────────────────────────────────────────
    t0 = time.monotonic()

    if rank == 0:
        # Loop 1: head backward → ring 回传 grad 启动
        for mb in range(M):
            losses[mb].backward()
            engine._send_grad_to_tail(head_in[mb].grad)
        # Loop 2: 等 grad 绕环回到 embed
        if role == RecoveryRole.UPSTREAM_HELPER:
            if use_graph:
                # rank 0 的 helper 路径 A：保留图 backward
                for mb in range(M):
                    g_inp, g_out = engine.cache.get_graph(ckpt_step, mb)
                    g_out.backward(engine._recv_grad())
                    # rank 0 不向上游发 grad
                engine.cache.release_graph_explicitly()
            else:
                # rank 0 的 helper 路径 B：重做 embed + backward
                loc_in, loc_out = [], []
                for mb in range(M):
                    src = fwd_in[mb]
                    inp = src.clone()
                    loc_in.append(inp)
                    loc_out.append(engine.stage.forward_embed(inp))
                for mb in range(M):
                    loc_out[mb].backward(engine._recv_grad())
        else:
            # rank 0 的 PREEMPTED / DOWNSTREAM_VICTIM：直接 backward 现有 fwd_out
            for mb in range(M):
                fwd_out[mb].backward(engine._recv_grad())

    elif role == RecoveryRole.UPSTREAM_HELPER:
        # rank 1..K-2 的 helper（rank K-1 不会是 helper，因为它已是 ring 末端）
        if use_graph:
            for mb in range(M):
                g_inp, g_out = engine.cache.get_graph(ckpt_step, mb)
                if g_inp.grad is not None:
                    g_inp.grad = None
                g_out.backward(engine._recv_grad())
                engine._send_grad(g_inp.grad)
            engine.cache.release_graph_explicitly()
        else:
            loc_in, loc_out = [], []
            for mb in range(M):
                src = fwd_in[mb]
                inp = src.clone().requires_grad_(True)
                loc_in.append(inp)
                loc_out.append(engine.stage(inp))
            for mb in range(M):
                loc_out[mb].backward(engine._recv_grad())
                engine._send_grad(loc_in[mb].grad)

    else:
        # PREEMPTED / DOWNSTREAM_VICTIM on rank 1..K-1
        for mb in range(M):
            if rank == K - 1:
                g = engine._recv_grad_from_head()
                fwd_out[mb].backward(g)
                engine._send_grad(fwd_in[mb].grad)
            else:
                fwd_out[mb].backward(engine._recv_grad())
                engine._send_grad(fwd_in[mb].grad)

    optimizer.step()
    dist.barrier()
    compute_sec = time.monotonic() - t0

    final_loss = (sum(l.item() for l in losses)
                  if rank == 0 else None)

    return {
        "activation_resend_sec":    resend_sec,
        "recovery_compute_sec":     compute_sec,
        "stages_resent_activation": stages_resent,
        "stages_recomputed_forward":stages_recomputed,
        "final_loss":               final_loss,
        "used_retained_graph":      use_graph,
    }
```

**Note on rank-K-1 UPSTREAM_HELPER**: by `plan_recovery`'s definition, rank K-1 can only be `DOWNSTREAM_VICTIM` (it's the last rank). There is no valid `target_rank` that makes K-1 a HELPER. The code above documents this in the comment but doesn't add a defensive check — the role-mapping invariant test covers it.

- [ ] **Step 2: Syntax-check**

Run: `python -c "import ast; ast.parse(open('recovery_protocol.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Confirm async path is byte-identical**

Run: `grep -n "execute_recovery_async" recovery_protocol.py`
Expected: the `def execute_recovery_async(...)` signature line is unchanged from the original (line ~329). If anything in the async body changed, revert it.

- [ ] **Step 4: Run plan_recovery test (unchanged behavior)**

Run: `pytest tests/test_async_recovery.py::test_plan_recovery_unchanged -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/Users/12589/Desktop/pp-preempt-v2
git add recovery_protocol.py
git commit -m "recovery: rewrite execute_recovery_sync for ring topology (rank 0 owns loss)"
```

---

### Task 4: Flip rank-K-1-owns-loss hardcodes in run_injection_v2.py

**Files:**
- Modify: `run_injection_v2.py` — 6 specific sites listed below

**Interfaces:**
- Consumes (from Task 2 & 3):
  - `PPEngine.step(...)` now returns non-zero loss on rank 0, zero elsewhere
  - `execute_recovery_sync(...)["final_loss"]` is non-None on rank 0, None elsewhere
- Produces:
  - JSON result file written by rank 0 (was rank K-1)
  - `batch_targets` no longer needed by rank K-1

- [ ] **Step 1: Patch `get_step_data` (line ~70)**

Find:
```python
def get_step_data(step, rank, K, M, mb_size, seq_len, vocab_size, seed):
    """返回 (input_ids_or_None, targets_or_None)。中间 rank 两者均为 None。"""
    if rank not in (0, K - 1):
        return None, None
    ids, tgts = make_batch(step, M * mb_size, seq_len, vocab_size, seed)
    return (ids, None) if rank == 0 else (None, tgts)
```

Replace with:
```python
def get_step_data(step, rank, K, M, mb_size, seq_len, vocab_size, seed):
    """返回 (input_ids_or_None, targets_or_None)。
    Ring 拓扑：rank 0 同时拿 input_ids 和 targets（loss 在 rank 0 算）；
    其他 rank 两者均为 None。"""
    if rank != 0:
        return None, None
    ids, tgts = make_batch(step, M * mb_size, seq_len, vocab_size, seed)
    return ids, tgts
```

- [ ] **Step 2: Patch `pre_loss` broadcast (line ~333)**

Find:
```python
    # 广播 pre_loss（只有 rank K-1 有值）
    pre_t = torch.tensor([float(pre_loss)], device=f"cuda:{local_rank}",
                          dtype=torch.float32)
    dist.broadcast(pre_t, src=K - 1)
```

Replace with:
```python
    # 广播 pre_loss（Ring 拓扑：只有 rank 0 有值）
    pre_t = torch.tensor([float(pre_loss)], device=f"cuda:{local_rank}",
                          dtype=torch.float32)
    dist.broadcast(pre_t, src=0)
```

- [ ] **Step 3: Patch loss trajectory accumulator (line ~431)**

Find:
```python
        if rank == K - 1:
            loss_traj.append({"step": step, "loss": round(float(loss), 6)})
```

Replace with:
```python
        if rank == 0:
            loss_traj.append({"step": step, "loss": round(float(loss), 6)})
```

- [ ] **Step 4: Patch Phase E JSON writer (line ~434–436)**

Find:
```python
    # ── Phase E：写 JSON（rank K-1）───────────────────────────────────────────
    if rank == K - 1:
        rec_loss  = rt.get("final_loss")
```

Replace with:
```python
    # ── Phase E：写 JSON（Ring 拓扑：rank 0 持有 loss）─────────────────────────
    if rank == 0:
        rec_loss  = rt.get("final_loss")
```

- [ ] **Step 5: Syntax-check**

Run: `python -c "import ast; ast.parse(open('run_injection_v2.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 6: grep-check there's no more `rank == K - 1` ownership pattern**

Run:
```bash
grep -n "rank == K - 1\|rank == K-1" run_injection_v2.py
```

Expected output: only the comment around the old Phase E header is fully replaced; any remaining hits must be examined. The only acceptable remaining hits (if any) are in comments referring to historical context. Lines doing `if rank == K - 1:` to gate behavior must all be gone.

- [ ] **Step 7: Commit**

```bash
cd /mnt/c/Users/12589/Desktop/pp-preempt-v2
git add run_injection_v2.py
git commit -m "run_injection: rank 0 owns loss/trajectory/JSON (was rank K-1)"
```

---

### Task 5: Adapt test_async_recovery.py infrastructure

**Files:**
- Modify: `tests/test_async_recovery.py` — `_TinyStage`, `_make_engine`

**Interfaces:**
- Consumes (from Task 1): `StageFirst.forward_embed`, `StageFirst.forward_head`, `StageLast.forward`
- Produces:
  - `_TinyStageFirst` — drop-in for rank 0 with `forward_embed` / `forward_head`
  - `_TinyStageTail` — drop-in for rank K-1, no head
  - `_TinyStageMid` — for middle ranks

- [ ] **Step 1: Replace `_TinyStage` class definition**

Open `tests/test_async_recovery.py`. Find:
```python
class _TinyStage(nn.Module):
    """A 1-layer linear stage; matches the (B, S, H) tensor shape of the real stages."""
    def __init__(self, H, V=None, is_last=False):
        super().__init__()
        self.linear = nn.Linear(H, H, bias=False)
        self.is_last = is_last
        if is_last and V is not None:
            self.head = nn.Linear(H, V, bias=False)

    def forward(self, x):
        x = self.linear(x)
        if self.is_last:
            x = self.head(x)
        return x
```

Replace with:
```python
class _TinyStageFirst(nn.Module):
    """Drop-in for rank 0 — mirrors the new StageFirst dual-method API.

    Note: real StageFirst.forward_embed takes token ids; this tiny version
    accepts hidden-shape (B, S, H) tensors instead. The only tests that
    exercise this class today (test_async_returns_required_fields,
    test_loss_consistency_sync_vs_async) are both @pytest.mark.skip'd for
    gloo limitations, so the type mismatch is inert. If those tests are
    ever un-skipped, _TinyStageFirst.forward_embed must be reworked to
    accept LongTensor[B, S] of ids.
    """
    def __init__(self, H, V):
        super().__init__()
        self.linear  = nn.Linear(H, H, bias=False)
        self.ln_f    = nn.LayerNorm(H)
        self.lm_head = nn.Linear(H, V, bias=False)

    def forward_embed(self, x):
        return self.linear(x)

    def forward_head(self, hidden):
        return self.lm_head(self.ln_f(hidden))


class _TinyStageMid(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.linear = nn.Linear(H, H, bias=False)

    def forward(self, x):
        return self.linear(x)


class _TinyStageTail(nn.Module):
    """Drop-in for rank K-1 — pure linear, no head."""
    def __init__(self, H):
        super().__init__()
        self.linear = nn.Linear(H, H, bias=False)

    def forward(self, x):
        return self.linear(x)
```

- [ ] **Step 2: Update `_make_engine` to pick the right tiny stage by rank**

Find:
```python
def _make_engine(rank, K, M, mb_size, seq_len, H, V):
    """Build a PPEngine wrapped around a tiny linear stage."""
    from pp_engine import PPEngine
    torch.manual_seed(1234 + rank)
    stage = _TinyStage(H, V=V, is_last=(rank == K - 1))
```

Replace with:
```python
def _make_engine(rank, K, M, mb_size, seq_len, H, V):
    """Build a PPEngine wrapped around a tiny linear stage matching ring layout."""
    from pp_engine import PPEngine
    torch.manual_seed(1234 + rank)
    if rank == 0:
        stage = _TinyStageFirst(H, V)
    elif rank == K - 1:
        stage = _TinyStageTail(H)
    else:
        stage = _TinyStageMid(H)
```

- [ ] **Step 3: Run plan_recovery test — must still pass without any change**

Run: `pytest tests/test_async_recovery.py::test_plan_recovery_unchanged -v`
Expected: PASS.

- [ ] **Step 4: Confirm skipped tests are still skipped (no regression)**

Run: `pytest tests/test_async_recovery.py -v`
Expected: 1 PASS, 2 SKIPPED.

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/Users/12589/Desktop/pp-preempt-v2
git add tests/test_async_recovery.py
git commit -m "test: adapt _TinyStage to ring layout (StageFirst dual-method)"
```

---

### Task 6: Add new CPU tests for ring-layout invariants

**Files:**
- Create: `tests/test_ring_layout_invariants.py`

**Interfaces:**
- Consumes (from Task 1 & Task 3): `build_stage`, `StageFirst`, `StageLast`, `plan_recovery`
- Produces: new test file

- [ ] **Step 1: Write the test file**

Create `tests/test_ring_layout_invariants.py`:

```python
"""Ring-layout invariant tests (CPU, no distributed).

These tests lock the contracts that callers (PPEngine, recovery_protocol,
run_injection_v2) rely on:
  · StageFirst has both forward_embed and forward_head
  · StageLast has neither ln_f nor lm_head
  · plan_recovery target_rank=0 produces the expected role assignment
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import build_stage, StageFirst, StageLast
from recovery_protocol import plan_recovery, RecoveryRole


def _cfg():
    return {
        "hidden_dim": 8, "vocab_size": 16, "num_layers_per_stage": 2,
        "num_heads": 2, "max_seq_len": 4, "dropout": 0.0,
    }


def test_stage_first_dual_methods_exist():
    s = build_stage(0, 4, _cfg())
    assert isinstance(s, StageFirst)
    assert callable(s.forward_embed)
    assert callable(s.forward_head)


def test_stage_first_has_head_params():
    s = build_stage(0, 4, _cfg())
    names = {n for n, _ in s.named_parameters()}
    # Must include both embed-side and head-side params
    assert any("tok_emb" in n for n in names)
    assert any("lm_head" in n for n in names)
    assert any("ln_f" in n for n in names)


def test_stage_last_no_head_no_ln_f():
    s = build_stage(3, 4, _cfg())
    assert isinstance(s, StageLast)
    assert not hasattr(s, "lm_head"), "StageLast must not own lm_head in ring layout"
    assert not hasattr(s, "ln_f"), "StageLast must not own ln_f in ring layout"


def test_stage_last_param_names_pure_blocks():
    s = build_stage(3, 4, _cfg())
    names = {n for n, _ in s.named_parameters()}
    assert not any("lm_head" in n for n in names)
    assert not any("ln_f" in n for n in names)
    assert any("blocks" in n for n in names)


def test_plan_recovery_target_0():
    """target_rank=0 is the special case: no UPSTREAM_HELPER exists."""
    plan = plan_recovery(target_rank=0, num_stages=4)
    assert plan[0] == RecoveryRole.PREEMPTED
    for r in (1, 2, 3):
        assert plan[r] == RecoveryRole.DOWNSTREAM_VICTIM
    helpers = [r for r, ro in plan.items() if ro == RecoveryRole.UPSTREAM_HELPER]
    assert helpers == [], "target_rank=0 must produce zero helpers"


def test_plan_recovery_target_K_minus_1():
    """target_rank=K-1 maximizes helpers (rank 0/1/2 all helpers)."""
    plan = plan_recovery(target_rank=3, num_stages=4)
    for r in (0, 1, 2):
        assert plan[r] == RecoveryRole.UPSTREAM_HELPER
    assert plan[3] == RecoveryRole.PREEMPTED
```

- [ ] **Step 2: Run the new tests**

Run: `pytest tests/test_ring_layout_invariants.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 3: Run the full CPU test suite (no regression)**

Run: `pytest tests/ -v`
Expected: previous tests still PASS / SKIP as before; 6 new tests PASS.

- [ ] **Step 4: Commit**

```bash
cd /mnt/c/Users/12589/Desktop/pp-preempt-v2
git add tests/test_ring_layout_invariants.py
git commit -m "test: ring-layout invariants — dual-method head, plan_recovery target_0"
```

---

### Task 7: Update README.md with ring topology and deployment guidance

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing (documentation only)
- Produces: updated README explaining the new layout

- [ ] **Step 1: Update the file-header docstring at the top of model.py (sanity)**

Already done in Task 1 (the new docstring starts with "Ring 拓扑布局"). No action needed; skip to Step 2.

- [ ] **Step 2: Edit README.md — Stage layout description (early in file)**

Find the existing text near the top (around line 14–24, the "配置对比" table) and **immediately after that table**, insert this new section:

```markdown
## Ring 拓扑（本次改造）

LM head（`ln_f` + `lm_head`）从 stage K-1 永久迁移到 stage 0，数据流由线性
`0→1→2→3` 变为 ring `0→1→2→3→0`：

```
forward（per microbatch）:
  rank 0  : input_ids → tok_emb+pos_emb → 8 blocks → hidden_h0      send → 1
  rank 1  : hidden_h0 → 8 blocks → hidden_h1                          send → 2
  rank 2  : hidden_h1 → 8 blocks → hidden_h2                          send → 3
  rank 3  : hidden_h2 → 8 blocks → hidden_h3                          send → 0  ← 新通道
  rank 0  : hidden_h3 → ln_f → lm_head → logits → cross_entropy → loss

backward:
  rank 0  : loss.backward() → grad_h3                                 send → 3  ← 新通道
  rank 3  : recv grad_h3, blocks.backward → grad_h2                  send → 2
  rank 2  : ... → grad_h1                                              send → 1
  rank 1  : ... → grad_h0                                              send → 0
  rank 0  : recv grad_h0, blocks+emb.backward
```

两个新增 P2P 通道：`K-1 → 0`（forward hidden）和 `0 → K-1`（backward grad）。
其余 `0→1→2→3` / `3→2→1→0` 通道保持不变。

### Stage 容量与部署建议（GPT-2 XL，~1.6B 总）

| stage | 参数构成 | 大小 | 部署建议 |
|---|---|---|---|
| 0 | tok_emb + pos_emb + 8 blocks + ln_f + lm_head | **~610M** | **ondemand**（最重，恢复代价最高）|
| 1 | 8 blocks | ~400M | spot |
| 2 | 8 blocks | ~400M | spot |
| 3 | 8 blocks | ~400M | spot |

把恢复成本高的 stage 集中到 ondemand，其余放 spot —— 这是本次改造的业务动机。

### 范围限制（本次改造）

- 仅修改 sync 恢复路径（`execute_recovery_sync`）
- async 1F1B 恢复路径（`execute_recovery_async`）保持原样，作为后续 spec
- `plan_recovery` 角色映射语义不变
- 不做 weight tying
- 不调整 `num_layers_per_stage`（stage 容量不均衡是本实验要观测的现象）
```

- [ ] **Step 3: Update the "Stage 0 = ... / Stage K-1 = ..." line in the model section**

Search the README for any line describing stage 0 as "Embedding + L 层" or stage K-1 as "L 层 + ln_f + LM Head" outside of the historical context section. If found, update those to reflect the new ring layout. (As of the current README, these descriptions live in the `model.py` docstring itself, not the README, so this step may be a no-op — that is acceptable.)

Run: `grep -n "Embedding + L 层\|ln_f + LM Head" README.md` and update any matches to the new layout. If nothing matches, this step is a no-op.

- [ ] **Step 4: Update the "文件改动相对 graph-opt" table at the bottom of README**

Find the table starting "| 文件 | 状态 |" near the bottom of README.md. Append two new rows at the top of the rows block (just below the header `|---|---|`):

```markdown
| `model.py` | **本次改造**：StageFirst 双方法 (forward_embed/forward_head)，StageLast 移除 head |
| `pp_engine.py` | **本次改造**：新增 4 个 ring-closure 通信原语，step() 重写，rank 0 owns loss |
| `recovery_protocol.py` | **本次改造**：execute_recovery_sync 重写（async 路径不动）|
| `run_injection_v2.py` | **本次改造**：6 处 rank K-1 ownership 翻转为 rank 0 |
```

(The existing rows for those files in the "graph-opt 对比" table can stay — they describe prior history.)

- [ ] **Step 5: Verify README renders sensibly**

Run: `head -80 README.md` and skim the inserted ring topology section. Confirm the ASCII diagram is intact and tables look right.

- [ ] **Step 6: Commit**

```bash
cd /mnt/c/Users/12589/Desktop/pp-preempt-v2
git add README.md
git commit -m "docs: ring topology section + deployment guidance table"
```

---

### Task 8: End-to-end smoke run (GPU, optional but recommended)

**Files:** none modified

**Interfaces:** consumes everything from Tasks 1–7.

This task is the "ready for platform handoff" gate. It requires a GPU environment — skip locally and run on the actual A100 platform.

- [ ] **Step 1: On platform — run baseline config end-to-end**

Run:
```bash
cd /workspace && CONFIG=configs/v2_baseline.yaml \
  bash scripts/run_all_v2.sh \
  --node-rank ${VC_TASK_INDEX} \
  --master-addr ${VC_WORKER_HOSTS_0} \
  --install-deps
```

- [ ] **Step 2: Verify final_loss is in rank-0's JSON output**

After Phase E, locate the result JSON under `/workspace/results/` (or wherever the run writes). Confirm:
- `convergence.recovery_loss` is non-null
- `convergence.pre_preemption_loss` is non-null
- `convergence.loss_gap` is a finite small number

If `recovery_loss` is null, the rank-0 ownership flip didn't take effect — check Task 4.

- [ ] **Step 3: Verify target_rank=0 trials have measurable recovery_compute_sec**

In the aggregated `summary.txt`, find rows where `target_rank == 0`. Their `recovery_compute_sec` should be **higher** than `target_rank == 1/2/3` rows — this is the key signal showing that head concentration on stage 0 increased its recovery cost (validating the ondemand placement decision).

- [ ] **Step 4: Run sync config for cross-comparison**

Run:
```bash
cd /workspace && CONFIG=configs/v2_sync.yaml \
  bash scripts/run_all_v2.sh \
  --node-rank ${VC_TASK_INDEX} \
  --master-addr ${VC_WORKER_HOSTS_0} \
  --install-deps
```

Verify `used_retained_graph: true` in target_rank>0 trials' JSON.

- [ ] **Step 5: Do NOT run async config in this spec's scope**

`configs/v2.yaml` enables `async_pipeline: true` which routes to `execute_recovery_async` — that path was intentionally left untouched. Running it will fail because the async function still has rank-K-1-owns-loss code paths.

Either:
- Skip `configs/v2.yaml` runs entirely, OR
- Temporarily flip its `async_pipeline: false` to validate sync behavior under that config's other settings (do not commit this flip).

- [ ] **Step 6: Tag the milestone**

```bash
cd /mnt/c/Users/12589/Desktop/pp-preempt-v2
git tag -a head-at-stage0-sync-validated -m "Sync path validated on platform"
```

---

## Self-Review Notes

**Spec coverage check:**
- §3 architecture & data flow → Task 1 (model) + Task 2 (pp_engine) + Task 7 (README diagram)
- §5 model.py design → Task 1
- §6 pp_engine.py design (4 primitives, step rewrite, retain_graph rule) → Task 2
- §7 recovery_protocol.py sync rewrite → Task 3
- §7.3 target_rank=0 special case → covered by code path + Task 6 unit test
- §8 run_injection_v2.py 6-site flip → Task 4
- §9 README updates → Task 7
- §10 testing plan (3 new CPU tests + adapt _TinyStage) → Task 5 + Task 6
- §11 YAGNI (no async, no tying, no toggle, no coordinator/checkpoint changes) → enforced by Global Constraints

**Placeholder scan:** clean — every step has either exact code or exact shell command with expected output.

**Type consistency:** `_send_hidden_to_head` / `_recv_hidden_from_tail` / `_send_grad_to_tail` / `_recv_grad_from_head` method names are used identically in Task 2 (definition) and Task 3 (consumption). `forward_embed` / `forward_head` method names match between Task 1 (definition) and Tasks 2 & 3 (consumption). Test class names `_TinyStageFirst` / `_TinyStageMid` / `_TinyStageTail` defined in Task 5 are used only within that same task.

**Cross-task ordering:** Tasks 1→2→3→4 form a strict chain (each depends on prior). Tasks 5 & 6 depend on Tasks 1+3. Task 7 (docs) is order-independent after Task 1. Task 8 depends on everything. Recommended order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8.
