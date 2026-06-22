# PP 抢占恢复代价实验 v4 — 1F1B 异步流水线调度（GPT-2 XL / A100 80G × 4 worker）

在 `pp-preempt-v2-graph-opt` 基础上,实现 PPT 第 4 页提出的"异步流水线调度":
**让 upstream helper 拿到 `grad_mb0` 时立刻 `graph.backward(mb0)`,与下游正在跑的
`forward(mb1)` 在墙钟上重叠**。

本包相对原 `pp-preempt-v2-async-1f1b` 的**唯一改动**是把模型规模放大到
GPT-2 XL 量级,并把 micro-batch 数从 4 提到 8,以便在 A100 80G × 4 worker 平台上
让 1F1B 的 `overlap_sec` 收益变得可观测(小模型 / 小 M 情况下重叠时间被
NCCL 启动开销盖住,几乎为 0)。

## 配置对比

| 字段 | 原 v2.yaml | **本包 v2.yaml** |
|---|---|---|
| `model.hidden_dim` | 1024 | **2048** |
| `model.num_heads` | 16 | 16(head_dim 128) |
| `model.num_layers_per_stage` | 4 | **8** |
| 总参数(4 stage 合计) | ~270M | **~1.6B(GPT-2 XL)** |
| `pp.num_microbatches` | 8 (old) / 4 (large 包) | **8** |
| `injection.inject_at_step` | 50 | 20 |
| 单 stage forward 量级 | ~5 ms | **~80–120 ms** |

每 stage 显存估算(fp32 + AdamW state + retained graph):≈ 14–15 GB,A100 80G 充裕。

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

## 与前三组的关系

| 包 | 恢复阶段调度 | 主要瓶颈 |
|---|---|---|
| `pp-preempt-v2-large` (基线 / activation 组) | 同步两阶段:先 resend 全部 → 再 compute 全部 | helper 的 backward 被下游 forward+loss 完全掩盖 |
| `pp-preempt-v2-graph-opt` | 同步两阶段 + functional_call 参数副本,Path A 可用 | 同上;graph 缓存节省的 forward 时间被下游计算掩盖 |
| `pp-preempt-v2-parallel-bwd` | Path C:所有 rank 缓存 grad_output,并行 backward | 训练阶段需缓存额外梯度 |
| **`pp-preempt-v2-async-1f1b` (本包)** | **1F1B per-microbatch 异步调度** | **接近理论下界:max(单 stage backward) + min 通信** |

## 三档对照(同一份代码,三份 config)

| Config | `retain_graph_interval` | `async_pipeline` | 等价机制 | 恢复路径 |
|---|---|---|---|---|
| `configs/v2_baseline.yaml` | **0** (关) | `false` | `pp-preempt-v2-large` | Path B(重做 forward + 同步) |
| `configs/v2_sync.yaml`     | 10        | `false` | `pp-preempt-v2-graph-opt` | Path A(保留图 + 同步) |
| `configs/v2.yaml`          | 10        | `true`  | `pp-preempt-v2-async-1f1b` 本体 | Path A + 1F1B 异步重叠 |

三份 config 模型规模(GPT-2 XL)完全一致,只差恢复机制开关。这样三轮跑出的结果
可以做完整横向比较:

- `baseline → sync`:看**保留图**省了多少 `recovery_compute_sec`(原 large 包讨论的图优化)
- `sync → async`:看 **1F1B 异步重叠**省了多少 `total_recovery_sec`(本包主角,`overlap_sec` 直接读)
- `baseline → async`:两层优化合起来的端到端收益

## Path A → 1F1B 对比

```
graph-opt(Path A 同步):
  rank0(helper):  send_act(0..7) ─────────────── recv_grad+bwd(0..7)
  rank1(helper):  recv+send(0..7) ──────────── recv_grad+bwd(0..7)+send_grad
  rank2(preempt): recv+fwd+send(0..7) ──────── recv_grad+bwd(0..7)+send_grad
  rank3(victim):  recv+fwd+loss(0..7) ─────── bwd(0..7)+send_grad
                  └── activation_resend ──┘└── recovery_compute ──┘

async-1f1b(本包):
  rank0(helper):  S0 S1 S2 ── B0(graph) ── B1 ── B2 ── B3 ── ...
  rank1(helper):  R0─S0 R1─S1 R2─S2 ── B0+SG0 ── B1+SG1 ── ...
  rank2(preempt): R0─F0─S0 R1─F1─S1 ── B0+SG0 ── B1+SG1 ── ...
  rank3(victim):  R0─F0─loss0 ── B0+SG0 R1─F1─loss1 ── B1+SG1 ...
                  ↑ resend 与 compute 在墙钟上重叠
```

## 结果中的新字段

```json
{
  "used_retained_graph":   true,
  "used_async_pipeline":   true,
  "wall_clock_breakdown": {
    "activation_resend_sec": 0.32,
    "recovery_compute_sec":  0.48,
    "overlap_sec":           0.18,   ← 关键:重叠了多少
    "total_recovery_sec":    0.95
  }
}
```

`overlap_sec = max(0, resend + compute - total)`。**模型放大后预期 overlap 显著 > 0**。

## 启动(平台 4 worker × 1 A100-80G-SXM4)

### 默认(async)

启动命令(平台填入):

```bash
cd /workspace && bash scripts/run_all_v2.sh \
  --node-rank ${VC_TASK_INDEX} \
  --master-addr ${VC_WORKER_HOSTS_0} \
  --install-deps
```

### 完整三档对照(推荐):提交三次任务

平台依次提交三个 job,启动命令唯一差异是 `CONFIG=...`:

**Job 1 — baseline(activation,等价 large 包)**:

```bash
cd /workspace && CONFIG=configs/v2_baseline.yaml \
  bash scripts/run_all_v2.sh \
  --node-rank ${VC_TASK_INDEX} \
  --master-addr ${VC_WORKER_HOSTS_0} \
  --install-deps
```

**Job 2 — graph-opt(保留图同步)**:

```bash
cd /workspace && CONFIG=configs/v2_sync.yaml \
  bash scripts/run_all_v2.sh \
  --node-rank ${VC_TASK_INDEX} \
  --master-addr ${VC_WORKER_HOSTS_0} \
  --install-deps
```

**Job 3 — async-1f1b(本包主角)**:

```bash
cd /workspace && CONFIG=configs/v2.yaml \
  bash scripts/run_all_v2.sh \
  --node-rank ${VC_TASK_INDEX} \
  --master-addr ${VC_WORKER_HOSTS_0} \
  --install-deps
```

三次产物分别打包到下载目录,文件名带 config tag,互不覆盖:
- `/workspace/downloads/async-1f1b_v2_baseline_<时间戳>.tar.gz`
- `/workspace/downloads/async-1f1b_v2_sync_<时间戳>.tar.gz`
- `/workspace/downloads/async-1f1b_v2_<时间戳>.tar.gz`

每个包内 `summary.txt` 含分组聚合表,可分别读取后画三组对比图。
`overlap_sec` 只在 Job 3(async)结果里 > 0。

### 只想跑两档

如果时间紧只想看 1F1B 增量收益,跳过 Job 1,只跑 Job 2 + Job 3 即可。
但**不要只跑 Job 3 单独**——没有 sync 基线就无法判断 `overlap_sec` 是否真的转化为 `total` 的下降。

## 测试(CPU,无需 GPU)

```bash
cd /workspace
pytest tests/test_async_recovery.py -v
```

三个测试:

1. `test_plan_recovery_unchanged` — 角色映射是 1F1B 改造的不变量
2. `test_async_returns_required_fields` — `execute_recovery_async` 返回正确字段
3. `test_loss_consistency_sync_vs_async` — 同 seed 下 sync 与 async 的
   `final_loss` 差 < 1e-5(数值等价)

## 显存 / 时长估算

- 单 stage 参数 ≈ 400M(fp32 ≈ 1.6 GB);AdamW state 2 × 1.6 = 3.2 GB;
  梯度 1.6 GB;8 个 mb activation cache ≈ 1 GB;retained graph + cloned params ≈ 4 GB;
  NCCL + tmp ≈ 2 GB。**合计 ≈ 14 GB / 80 GB**,有大量富余。
- 单 trial 含 19 步 warmup + Phase B/C/D 全过程,GPT-2 XL 量级下约 1–3 分钟;
  4 stages × 3 repeats = 12 trials,**总耗时单次 config ≈ 20–40 分钟**;
  跑完 sync + async 两轮 ≈ 1–1.5 小时。

## 文件改动相对 graph-opt

| 文件 | 状态 |
|---|---|
| `model.py` | **本次改造**：StageFirst 双方法 (forward_embed/forward_head)，StageLast 移除 head |
| `pp_engine.py` | **本次改造**：新增 4 个 ring-closure 通信原语，step() 重写，rank 0 owns loss |
| `recovery_protocol.py` | **本次改造**：execute_recovery_sync 重写（async 路径不动）|
| `run_injection_v2.py` | **本次改造**：6 处 rank K-1 ownership 翻转为 rank 0 |
| `analyze_v2.py` | 小改:新增 `overlap_sec` 列、`async_used` 列 |
| `configs/v2.yaml` | **本包改动**:放大到 GPT-2 XL 量级;`async_pipeline: true` |
| `configs/v2_sync.yaml` | **本包新增**:三档对照之 graph-opt(`async_pipeline: false`,图保留 ON) |
| `configs/v2_baseline.yaml` | **本包新增**:三档对照之 activation baseline(`retain_graph_interval: 0`,图保留 OFF) |
| `scripts/run_all_v2.sh` | **本包小改**:`CONFIG` 环境变量支持,包名含 config tag 不互相覆盖,trial 前清旧 results |
| `requirements.txt` | 新增 `pytest>=7.0` |
| `tests/test_async_recovery.py` | 新增 |
| 其余 (model/cache/checkpoint/coordinator/pp_engine) | 不变 |
