"""
差异化恢复协议。

plan_recovery(target_rank, num_stages) → {rank: RecoveryRole}

execute_recovery_sync(plan, engine, ckpt_step, ...) → timing dict

恢复角色：
  UPSTREAM_HELPER   : ranks < target_rank
    → forward 阶段：发送 cached detached activation
    → backward 阶段：若 has_graph → 直接用保留图 backward（跳过重做 forward）
                     否则 → 重做 forward + backward（退化路径）

  PREEMPTED         : target_rank
    → 重做 forward + backward（相当于重新执行 inject_step）

  DOWNSTREAM_VICTIM : ranks > target_rank
    → 重做 forward + backward
"""
import time
from collections import deque
from enum import Enum

import torch
import torch.distributed as dist
import torch.nn.functional as F

from pair_groups import ensure_pair_groups, pair_group, pair_send, pair_recv


class RecoveryRole(Enum):
    UPSTREAM_HELPER   = "upstream_helper"
    PREEMPTED         = "preempted"
    DOWNSTREAM_VICTIM = "downstream_victim"


def plan_recovery(target_rank: int, num_stages: int) -> dict:
    plan = {}
    for r in range(num_stages):
        if r < target_rank:
            plan[r] = RecoveryRole.UPSTREAM_HELPER
        elif r == target_rank:
            plan[r] = RecoveryRole.PREEMPTED
        else:
            plan[r] = RecoveryRole.DOWNSTREAM_VICTIM
    return plan


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

    # Bootstrap pair sub-PGs on the FRESH default PG (Phase B just rebuilt it).
    # engine.step() also calls this, but during recovery the PG is new,
    # so the cache invalidates and rebuilds here — must run before any P2P.
    # Collective on default PG → all ranks must call in lock-step.
    ensure_pair_groups(K, engine.device)

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



# ═══════════════════════════════════════════════════════════════════════════
# 1F1B async recovery (PPT 第 4 页方案) — Bamboo-style 2-rank sub-PG transport
#
# 把"先 resend 全部 mb → 再 compute 全部 mb"两阶段串行,改成 per-mb 1F1B 调度。
#
# 通信原语:借鉴 Bamboo (NSDI'23) 的 p2p.py 实现
#   uclasystem/bamboo:external/deepspeed/deepspeed/runtime/pipe/p2p.py
# 关键设计:
#   1) 在重建 default PG 之后,为每条通信边单独建 dist.new_group:
#      相邻边 [i, i+1] 加上 ring closure [0, K-1]
#      → 每个 pair 拥有独立的 NCCL communicator,不受 default PG eager-init
#      mode 下 P2P sub-comm 创建竞争的影响。
#   2) 用 dist.broadcast(tensor, src_rank, group=pair_group) 模拟 P2P:
#      broadcast 是集合操作,NCCL 在 sub-PG 上调度它从不死锁。
#   3) 每个 pair_group 在 new_group 后立刻做一次 warm-up broadcast,强制
#      NCCL 真正建好 sub-comm,后续业务调用零延迟。
#   4) 每次 broadcast 之后 async barrier + 60s timeout,失败可被检测,
#      避免 17 分钟无响应。
#
# 为什么这同时解决了 isend 死锁和阻塞 1F1B chicken-egg:
#   · isend 死锁:不用 isend,改用 broadcast on sub-PG(没有 lazy sub-comm)
#   · 1F1B chicken-egg:broadcast 是同步集合,但每个 sub-PG 只 2 个 rank,
#     send 端的 broadcast 跟 recv 端的 broadcast 严格配对,无死锁路径
#
# 数值不变量:每个 mb 的张量来源/去向、graph 使用方式与 sync 版完全一致。
# ═══════════════════════════════════════════════════════════════════════════

# Pair-sub-PG creation + P2P helpers moved to pair_groups.py so training
# (pp_engine.py) shares one warmed-up cache with recovery. Async recovery's
# extra requirements — async_op send handles + explicit drain at end —
# stay here as thin wrappers around pair_groups.pair_send/pair_recv.

# In-flight async sends; all are .wait()ed at end of recovery to ensure
# their buffers (especially cloned grads) outlive the NCCL transfer.
_PENDING_SENDS: list = []


def _send_act(engine, x, dst):
    """
    Async send via pair sub-PG. Handle is queued for end-of-recovery drain.
    Async is REQUIRED for 1F1B: rank K-1 must issue send without waiting
    for K-2 to recv, so it can proceed to the next forward.
    """
    work = pair_send(x.contiguous(), dst=dst, async_op=True)
    _PENDING_SENDS.append(work)


def _recv_act(engine, src):
    buf = torch.empty((engine.mb_size, engine.seq_len, engine.H),
                      device=engine.device, dtype=torch.float32)
    pair_recv(buf, src=src)
    return buf


def _send_grad(engine, g, dst):
    # Clone before async send: the .grad field may be cleared by the next
    # backward before NCCL finishes transmitting.
    work = pair_send(g.detach().clone().contiguous(), dst=dst, async_op=True)
    _PENDING_SENDS.append(work)


def _recv_grad(engine, src):
    buf = torch.empty((engine.mb_size, engine.seq_len, engine.H),
                      device=engine.device, dtype=torch.float32)
    pair_recv(buf, src=src)
    return buf


def _send_hidden_to_head(engine, x):
    _send_act(engine, x, dst=0)


def _recv_hidden_from_tail(engine):
    return _recv_act(engine, src=engine.K - 1)


def _send_grad_to_tail(engine, g):
    _send_grad(engine, g, dst=engine.K - 1)


def _recv_grad_from_head(engine):
    return _recv_grad(engine, src=0)


def _drain_pending_sends():
    """Wait for all in-flight async sends to complete; clear queue."""
    global _PENDING_SENDS
    for work in _PENDING_SENDS:
        try:
            work.wait()
        except Exception:
            pass
    _PENDING_SENDS = []


def execute_recovery_async(plan, engine, ckpt_step, resume_step,
                           batch_input_ids, batch_targets, optimizer) -> dict:
    """
    1F1B async recovery. Drop-in replacement for execute_recovery_sync.

    Returns the same dict shape plus two new keys:
      - "used_async_pipeline": True
      - "overlap_sec": max(0, t_resend + t_compute - t_total)
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

    # Bamboo-style: pre-create + warm-up every pipeline-edge sub-PG,
    # including the K-1 → 0 / 0 → K-1 ring-closure edge.
    # All ranks call this in lock-step — it's a collective op sequence.
    # Shared with training (pp_engine.step) and sync recovery via pair_groups.
    ensure_pair_groups(K, engine.device)
    # Defensive: clear any send-handles left over from a prior trial.
    _drain_pending_sends()

    # Per-rank 1F1B warm-up depth (classic 1F1B).
    num_warmup = min(K - 1 - rank, M)

    # Per-mb context: stores tensors needed by the matching backward.
    #   payload kinds (first element of tuple):
    #     "graph"           helper has retained graph → (g_inp, g_out)
    #     "recompute"       helper no graph → (loc_in, loc_out)
    #     "preempt0"        rank-0 preempted   → (inp, out)
    #     "preempt_mid"     middle preempted   → (fwd_in, fwd_out)
    #     "preempt_last"    rank K-1 preempted → (fwd_in, fwd_out)
    #     "downstream_mid"  middle downstream  → (fwd_in, fwd_out)
    #     "downstream_last" rank K-1 downstrm  → (fwd_in, fwd_out)
    fwd_ctx: dict = {}
    losses: list = []                 # rank 0 accumulates head/loss here

    # Wall-clock tracking
    t_total_start = time.monotonic()
    t_first_send_done: list[float] = []
    t_last_send_done:  list[float] = []
    t_compute_start:   list[float] = []
    t_compute_end:     list[float] = []

    def _mark_send():
        now = time.monotonic()
        if not t_first_send_done:
            t_first_send_done.append(now)
        t_last_send_done.clear()
        t_last_send_done.append(now)

    def _mark_compute():
        now = time.monotonic()
        if not t_compute_start:
            t_compute_start.append(now)
        t_compute_end.clear()
        t_compute_end.append(now)

    # ── per-mb forward ───────────────────────────────────────────────────────

    def do_forward(mb):
        if role == RecoveryRole.UPSTREAM_HELPER:
            # Drain the symmetric recv (sync version does: `_ = _recv_act()`)
            if rank > 0:
                _ = _recv_act(engine, src=rank - 1)
            if use_graph:
                g_inp, g_out = engine.cache.get_graph(ckpt_step, mb)
                fwd_ctx[mb] = ("graph", g_inp, g_out)
                _, cached_out = engine.cache.get(ckpt_step, mb)
                _send_act(engine, cached_out, dst=rank + 1)
                _mark_send()
            else:
                # Path B: no graph. SEND the cached_out downstream (matches sync
                # semantics — receiver sees the cached tensor), keep a locally
                # recomputed `out` to backward on.
                cached_inp, cached_out = engine.cache.get(ckpt_step, mb)
                inp = cached_inp.clone() if rank == 0 else cached_inp.clone().requires_grad_(True)
                out = engine.stage.forward_embed(inp) if rank == 0 else engine.stage(inp)
                fwd_ctx[mb] = ("recompute", inp, out)
                _send_act(engine, cached_out, dst=rank + 1)
                _mark_send()

        elif role == RecoveryRole.PREEMPTED:
            if rank == 0:
                s = mb * engine.mb_size
                inp = batch_input_ids[s:s + engine.mb_size].to(engine.device)
                out = engine.stage.forward_embed(inp)
                fwd_ctx[mb] = ("preempt0", inp, out)
                _send_act(engine, out, dst=rank + 1)
                _mark_send()
            elif rank == K - 1:
                inp = _recv_act(engine, src=rank - 1); inp.requires_grad_(True)
                out = engine.stage(inp)
                fwd_ctx[mb] = ("preempt_last", inp, out)
                _send_hidden_to_head(engine, out)
                _mark_send()
            else:
                inp = _recv_act(engine, src=rank - 1); inp.requires_grad_(True)
                out = engine.stage(inp)
                fwd_ctx[mb] = ("preempt_mid", inp, out)
                _send_act(engine, out, dst=rank + 1)
                _mark_send()

        else:  # DOWNSTREAM_VICTIM
            if rank == K - 1:
                inp = _recv_act(engine, src=rank - 1); inp.requires_grad_(True)
                out = engine.stage(inp)
                fwd_ctx[mb] = ("downstream_last", inp, out)
                _send_hidden_to_head(engine, out)
                _mark_send()
            else:
                inp = _recv_act(engine, src=rank - 1); inp.requires_grad_(True)
                out = engine.stage(inp)
                fwd_ctx[mb] = ("downstream_mid", inp, out)
                _send_act(engine, out, dst=rank + 1)
                _mark_send()

    # ── per-mb backward ──────────────────────────────────────────────────────

    def do_backward(mb):
        kind = fwd_ctx[mb][0]

        if rank == 0:
            hidden = _recv_hidden_from_tail(engine)
            hidden.requires_grad_(True)
            logits = engine.stage.forward_head(hidden)
            s = mb * engine.mb_size
            tgt = batch_targets[s:s + engine.mb_size].to(engine.device)
            loss = F.cross_entropy(logits.view(-1, engine.V), tgt.view(-1)) / M
            losses.append(loss)
            loss.backward()
            _send_grad_to_tail(engine, hidden.grad)
            _mark_send()

        if kind == "graph":
            _, g_inp, g_out = fwd_ctx[mb]
            grad_out = _recv_grad(engine, src=rank + 1)
            if rank > 0 and g_inp.grad is not None:
                g_inp.grad = None
            g_out.backward(grad_out)
            if rank > 0:
                _send_grad(engine, g_inp.grad, dst=rank - 1)
                _mark_send()
            # cloned-params .grad is intentionally discarded (helper exists to
            # deliver g_inp.grad upstream, matches sync)

        elif kind == "recompute":
            _, loc_in, loc_out = fwd_ctx[mb]
            grad_out = _recv_grad(engine, src=rank + 1)
            loc_out.backward(grad_out)
            if rank > 0:
                _send_grad(engine, loc_in.grad, dst=rank - 1)
                _mark_send()

        elif kind == "preempt0":
            _, inp, out = fwd_ctx[mb]
            grad_out = _recv_grad(engine, src=rank + 1)
            out.backward(grad_out)
            # rank 0 doesn't send grad upstream

        elif kind in ("preempt_mid", "downstream_mid"):
            _, fwd_in, fwd_out = fwd_ctx[mb]
            grad_out = _recv_grad(engine, src=rank + 1)
            fwd_out.backward(grad_out)
            _send_grad(engine, fwd_in.grad, dst=rank - 1)
            _mark_send()

        elif kind in ("preempt_last", "downstream_last"):
            _, inp, out = fwd_ctx[mb]
            grad_out = _recv_grad_from_head(engine)
            out.backward(grad_out)
            _send_grad(engine, inp.grad, dst=rank - 1)
            _mark_send()

        else:
            raise RuntimeError(f"unknown fwd_ctx kind: {kind}")

        del fwd_ctx[mb]
        _mark_compute()

    # IMPORTANT: 1F1B with blocking pair-broadcast can deadlock on the
    # chicken-egg if every rank tries to push max forwards before any
    # backward. To avoid this, we use the OFFSET 1F1B schedule (Megatron-
    # style): rank r does (K-1-r) warm-up forwards, then alternates F/B,
    # then drains backwards. The asymmetric warm-up depths break the
    # circular wait because rank K-1's "warm-up" is 0 — it immediately
    # starts F→B→F→B and produces the first grad as soon as mb 0's
    # forward reaches it.

    # ── Phase 1: warm-up (push num_warmup forwards to fill the pipeline) ────
    for mb in range(num_warmup):
        do_forward(mb)

    # ── Phase 2: 1F1B steady state ──────────────────────────────────────────
    for mb in range(num_warmup, M):
        do_forward(mb)
        bwd_mb = mb - num_warmup
        do_backward(bwd_mb)

    # ── Phase 3: cool-down (drain remaining backwards) ──────────────────────
    for bwd_mb in range(M - num_warmup, M):
        do_backward(bwd_mb)

    # All forwards/backwards issued; wait for every async send to finish
    # before touching params (optimizer.step) or barriering.
    _drain_pending_sends()

    if use_graph:
        engine.cache.release_graph_explicitly()

    optimizer.step()
    dist.barrier()
    t_total = time.monotonic() - t_total_start

    # Wall-clock breakdown (per rank).
    if t_first_send_done and t_last_send_done:
        resend_sec = max(0.0, t_last_send_done[-1] - t_total_start)
    else:
        resend_sec = 0.0
    if t_compute_start and t_compute_end:
        compute_sec = max(0.0, t_compute_end[-1] - t_compute_start[0])
    else:
        compute_sec = 0.0
    overlap_sec = max(0.0, resend_sec + compute_sec - t_total)

    final_loss = (sum(l.item() for l in losses)
                  if rank == 0 else None)

    return {
        "activation_resend_sec":    resend_sec,
        "recovery_compute_sec":     compute_sec,
        "stages_resent_activation": stages_resent,
        "stages_recomputed_forward":stages_recomputed,
        "final_loss":               final_loss,
        "used_retained_graph":      use_graph,
        "used_async_pipeline":      True,
        "overlap_sec":              overlap_sec,
    }
