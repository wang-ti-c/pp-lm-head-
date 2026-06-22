# 把 LM head 从 stage 3 迁移到 stage 0（Ring 拓扑改造）

**日期**：2026-06-22
**范围**：`pp-preempt-v2`，sync 恢复路径
**状态**：设计已确认，待写实现计划

---

## 1. 目标

把 LM head（`ln_f` + `lm_head`）以及 cross-entropy loss 计算从 rank K-1 永久迁移到 rank 0。

**业务动机**：在云上部署时，把"恢复成本高的 stage"集中到 ondemand（可靠节点），其余 stage 放到 spot（便宜但可被回收）。LM head 的参数量（~103M @ GPT-2 XL）显著大于单个 transformer block，把它和 embedding 表并到 rank 0 之后：

| stage | 改前 | 改后 |
|---|---|---|
| 0 | tok_emb + pos_emb + 8 blocks ≈ **505M** | + ln_f + lm_head ≈ **610M**（最重，放 ondemand）|
| 1 | 8 blocks ≈ 400M | 400M |
| 2 | 8 blocks ≈ 400M | 400M |
| 3 | 8 blocks + ln_f + lm_head ≈ **503M** | 8 blocks ≈ **400M**（最轻，放 spot）|

总参数量几乎不变；变化只是 ~103M 在 stage 间的位置。

## 2. 范围

**本 spec 只覆盖 sync 恢复路径**（`execute_recovery_sync`）。async 1F1B 路径（`execute_recovery_async`）作为独立 follow-up。

**完全迁移，不保留旧布局开关**。旧的 baseline / sync 三档实验结果失效，新拓扑下重跑。

## 3. 架构与数据流

数据流从线性 `0→1→2→3` 变为 ring：

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

两个新 P2P 通道：`K-1 → 0`（forward 时的 hidden_h3）和 `0 → K-1`（backward 时的 grad_h3）。其余通道不变。

## 4. 文件改动总览

| 文件 | 改动 |
|---|---|
| `model.py` | 重写 `StageFirst` 与 `StageLast`；新增 `forward_embed` / `forward_head` 方法 |
| `pp_engine.py` | 新增 4 个通信原语；重写 `step()`；调整 `do_retain_here` 规则 |
| `recovery_protocol.py` | 重写 `execute_recovery_sync` 中 rank 0 / rank K-1 的 forward/backward 分支；翻转 `final_loss` 来源 |
| `run_injection_v2.py` | 把 6 处 `rank K-1` 硬编码翻转为 `rank 0`；调整数据加载条件 |
| `configs/*.yaml` | 不改（模型超参不变） |
| `README.md` | 更新 stage 描述、新增 ring 拓扑图、新增容量与部署建议表 |
| `tests/test_async_recovery.py` | `_TinyStage` 适配新接口；新增 3 个 CPU 测试 |

## 5. model.py 详细设计

```python
class StageFirst(nn.Module):
    """Stage 0: 双重职责。Forward 拆成两个具名方法，对应 ring 上两个物理时机。"""
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
        """microbatch 起始：发给 rank 1 之前的整段计算。"""
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x    = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward_head(self, hidden):
        """microbatch 末尾：从 rank K-1 收到 hidden 后算 logits。"""
        return self.lm_head(self.ln_f(hidden))


class StageLast(nn.Module):
    """Stage K-1: 退化为纯 transformer blocks，结构等同 StageMiddle。
    保留独立类是为 build_stage() 的角色身份清晰。"""
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
```

**`StageFirst.forward()` 被删除**：调用方必须显式选择 `forward_embed` 或 `forward_head`。两个时机在物理上完全不同（一个是 mb 起始，一个是 mb 末尾），用方法名表达比用 flag 参数更难写错。

**weight tying**：本 spec 不做。引入 tying 是独立优化点。

## 6. pp_engine.py 详细设计

### 6.1 新增通信原语

```python
def _send_hidden_to_head(self, x):
    """rank K-1 → rank 0：环上的回传通道（forward 时）"""
    dist.send(x.contiguous(), dst=0)

def _recv_hidden_from_tail(self):
    """rank 0 接收 rank K-1 发回的 hidden（forward 末段）"""
    buf = torch.empty((self.mb_size, self.seq_len, self.H),
                      device=self.device, dtype=torch.float32)
    dist.recv(buf, src=self.K - 1)
    return buf

def _send_grad_to_tail(self, g):
    """rank 0 → rank K-1：环上的回传通道（backward 时）"""
    dist.send(g.contiguous(), dst=self.K - 1)

def _recv_grad_from_head(self):
    """rank K-1 接收 rank 0 发回的 grad（backward 启动）"""
    buf = torch.empty((self.mb_size, self.seq_len, self.H),
                      device=self.device, dtype=torch.float32)
    dist.recv(buf, src=0)
    return buf
```

### 6.2 step() 重写

两段式调度（保持与原 `step()` 同结构：所有 mb 先 forward，再所有 mb backward）：

**Forward 阶段（per mb）**：

- rank 0：`out_h0 = stage.forward_embed(inp)`，cache 保存 `(inp, out_h0)`，`_send_act(out_h0) → rank 1`。**不算 loss**。
- rank 1, 2：同原版（recv → stage(inp) → send）。
- rank K-1：`out_h3 = stage(inp)`，`_send_hidden_to_head(out_h3) → rank 0`，把 `(inp, out_h3)` 存入 `fwd_in / fwd_out` 列表（与原 middle stage 完全相同的数据结构，无需新结构）。

**Loss 阶段（rank 0，per mb）**：

```python
for mb in range(M):
    hidden_h3 = self._recv_hidden_from_tail()
    hidden_h3.requires_grad_(True)
    logits = self.stage.forward_head(hidden_h3)
    tgt = batch_targets[mb*mb_size:(mb+1)*mb_size].to(device)
    losses.append(F.cross_entropy(logits.view(-1, V), tgt.view(-1)) / M)
    head_ctx[mb] = hidden_h3
```

**Backward 阶段（per mb）**：

- rank 0：分两个串行循环
  - **循环 1（per mb）** —— 触发 head backward，启动环回：
    1. `losses[mb].backward()`
    2. `_send_grad_to_tail(head_ctx[mb].grad)`
  - **循环 2（per mb）** —— 等 grad 绕环回到 embed：
    1. `g = _recv_grad()` 从 rank 1 收
    2. `fwd_out_embed[mb].backward(g, retain_graph=do_retain_here)`
    3. `transfer_grads(cloned)`
- rank K-1：`g = _recv_grad_from_head()`，`fwd_out[mb].backward(g)`，`_send_grad(fwd_in[mb].grad) → rank K-2`（与中间 stage 同形态）
- rank 1, 2：同原版

### 6.3 retain_graph 规则调整

原规则：`do_retain_here = do_retain and (self.rank < self.K - 1)`，例外是因为旧 rank K-1 自己算 loss + backward 不需保留图给别人。

新规则：`do_retain_here = do_retain`（去掉例外）。理由：

- rank K-1 现在与中间 stage 同性质（forward 完 send，等 grad 回来才 backward），所以也需要保留图供 UPSTREAM_HELPER 路径复用
- rank 0 的 **embed 段**仍走原 `_forward_with_clones` 路径保留图（它是 helper）
- rank 0 的 **head 段** 不走 cloned-params 路径：它是 loss 的本地终点，backward 走 stage 原始参数；`retain_graph` 在 head 段无意义（每个 mb 一次性 backward 完即释放）

## 7. recovery_protocol.py 详细设计（仅 sync）

### 7.1 plan_recovery 不变

按 rank 顺序的角色映射（UPSTREAM_HELPER / PREEMPTED / DOWNSTREAM_VICTIM）语义保持不变。`test_plan_recovery_unchanged` 继续通过。

### 7.2 execute_recovery_sync 改动

**rank 0 行为**：

- 角色是 PREEMPTED 或 DOWNSTREAM_VICTIM 时：
  - forward 段：`out_h0 = stage.forward_embed(inp); _send_act(out_h0)`；记录 `fwd_in_embed[mb] / fwd_out_embed[mb]`
  - loss 段：循环 `_recv_hidden_from_tail()` → `forward_head` → 算 loss → 存 `head_ctx[mb]`
  - backward 段：先 `losses[mb].backward(); _send_grad_to_tail(head_ctx[mb].grad)`，再 `_recv_grad()` → `fwd_out_embed[mb].backward(g)`
- 角色是 UPSTREAM_HELPER 时（这只在 target_rank != 0 时出现）：
  - resend cached act 给 rank 1（与原版相同）
  - 但**还要兼任 head 计算者**：等 rank K-1 把 hidden 发回，算 loss，backward，回送 grad 给 rank K-1
  - 这是 ring 拓扑下 rank 0 的固有双重身份

**rank K-1 行为**：

- 不再算 loss、不再持有 `losses`
- forward 段：`out_h3 = stage(inp); _send_hidden_to_head(out_h3)`；记录 `fwd_in[mb] / fwd_out[mb]`（与中间 stage 同形态）
- backward 段：`g = _recv_grad_from_head()` → `fwd_out[mb].backward(g)` → `_send_grad(fwd_in[mb].grad) → K-2`

**rank 1 / 2 行为**：与原版完全相同（中间 stage 不感知 head 在哪里）。

**返回值**：

```python
final_loss = (sum(l.item() for l in losses) if rank == 0 else None)
```

### 7.3 target_rank=0 的特殊情况

rank 0 被抢占时没有 UPSTREAM_HELPER（按 plan 定义），rank 1/2/3 都是 DOWNSTREAM_VICTIM。这种 trial 的恢复成本最高（要重做 embed + head + loss + 整个 backward 链），实验上**正好**凸显"head 集中到 stage 0 之后 stage 0 的恢复代价"——这是验证"放 ondemand 是合理选择"的关键数据点。**不在 plan_recovery 里排除此情况**。

### 7.4 UPSTREAM_HELPER 的 cache 不变

helper 仍然 resend `cached_out` 给下游，与 head 在哪里无关。`activation_cache.py` 不动。

## 8. run_injection_v2.py 改动

把所有 "rank K-1 owns loss" 的硬编码翻转为 rank 0：

| 位置 | 改动 |
|---|---|
| L70 `if rank not in (0, K-1):` 数据加载 skip | 改为 `if rank != 0:`，rank K-1 不再需要 `batch_targets` |
| L327 `pre_loss = engine.step(...)` | 行为变了：rank 0 现在返回非零 loss，rank K-1 返回 0 |
| L333 `dist.broadcast(pre_t, src=K-1)` | 改为 `src=0` |
| L431 `if rank == K - 1: loss_traj.append(...)` | 改为 `if rank == 0:` |
| L435 `if rank == K - 1: ... 写 JSON` | 改为 `if rank == 0:` |
| L436 `rt.get("final_loss")` | 在 rank 0 的 ret 上拿（recovery_protocol 已对应翻转）|

抢占注入逻辑（`coordinator.py` / `inject_at_step`）**不改**——抢占的是 stage 进程，与 loss 在哪里无关。

## 9. configs 与 README

**configs/*.yaml**：模型超参（hidden_dim / num_layers_per_stage / vocab_size）完全不改。三档对照 baseline/sync/async 的开关含义不变。

**README.md** 更新：

- 顶部 stage 描述：
  - 旧 "Stage 0 = Embedding + L 层 / Stage K-1 = L 层 + ln_f + LM Head"
  - 新 "Stage 0 = Embedding + L 层 + ln_f + LM Head（双向）/ Stage K-1 = L 层"
- 新增小节 "Ring 拓扑与新通信通道"，画出 §3 的数据流图
- 新增小节 "Stage 容量与部署建议"，给出 §1 的参数量表，明确：rank 0 → ondemand；rank 1/2/3 → spot
- "文件改动相对 graph-opt" 表更新：model.py / pp_engine.py / recovery_protocol.py 标记为"本次改动"

## 10. 测试

**CPU 单元测试（pytest）**：

- `test_plan_recovery_unchanged` —— 保持不动
- `test_plan_recovery_unchanged_with_target_0` —— 新增。显式 `target_rank=0` 时，rank 0 是 PREEMPTED，其他都是 DOWNSTREAM_VICTIM
- `test_stage_first_dual_forward` —— 新增。构造 `StageFirst`，分别调用 `forward_embed(input_ids)` 和 `forward_head(hidden)`，断言输出形状 `(B, T, H)` 与 `(B, T, V)`，断言 `forward_head` 的反向梯度能流到 `lm_head.weight`
- `test_stage_last_no_head` —— 新增。`build_stage(K-1, K, cfg)` 返回的实例不应有 `lm_head` 或 `ln_f` 属性，防止旧代码残留

**`_TinyStage` 适配**：去掉 `is_last` 路径里的 `self.head`。新增一个 `_TinyStageFirst`，带 `forward_embed` / `forward_head` 两个方法，模拟新 rank 0。

**两个 async 测试保持 skip**（gloo 限制未变）。

**GPU 平台验收**：

- 跑 `v2_baseline.yaml` + `v2_sync.yaml`
- 验证 `final_loss` 字段在 rank 0 的 JSON 里出现
- 验证 `loss_gap` 收敛行为与旧拓扑同数量级（同 seed/优化器/模型容量，差异只来自数值顺序）
- 验证 `target_rank=0` trial 的 `recovery_compute_sec` 显著大于其他 trial—— 这是核心结果

## 11. 不做的事（YAGNI）

- 不做 weight tying
- 不改 `execute_recovery_async`（async 路径作为独立 follow-up spec）
- 不动 `coordinator.py` / `checkpoint_v2.py` / `activation_cache.py` / `analyze_v2.py`
- 不在 config 里加 "head_at_stage0" 开关（完全迁移，无运行时切换需求）
- 不在本次改动里调整 `num_layers_per_stage` 平衡 stage 容量（容量不均衡是本实验**想观测**的现象，不是要消除的问题）
