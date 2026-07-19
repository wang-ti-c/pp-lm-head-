"""
PP 抢占恢复代价实验 v2

完全自包含，无任何外部文件依赖：
  - 数据：在代码内部用 torch.Generator 合成随机 token 序列
  - 模型：从随机初始化开始训练
  - 无需任何 checkpoint 文件、数据集文件

实验流程：
  Phase A   step 1 → inject_step-1    从零 warmup，累积 activation cache
  Phase A.6 step inject_step           填充 cache + 保留 autograd 图（若 inject_step % retain_graph_interval == 0）
  Phase A.7                            model + activation cache → 磁盘（供 target 重启后恢复）
  Phase B                              真实 PG 摧毁 + 重建（文件协调，tcp:// 重建）
                                       detection_sec / pg_destroy_sec / ckpt_load_sec / comm_rebuild_sec 全为真实测量
  Phase C                              差异化恢复（upstream resend / preempted+downstream recompute）
  Phase D   inject_step+1 → +catch_up 继续训练，输出 loss 轨迹
  Phase E                              rank K-1 写 JSON

平台：1 worker × 4 replicas
  VC_TASK_INDEX      → torchrun --node_rank
  VC_WORKER_HOSTS_0  → --master-addr（训练 rendezvous + Phase B TCP 地址）

启动命令（4 个节点完全相同）：
  cd /workspace
  bash scripts/run_all_v2.sh \\
    --node-rank  ${VC_TASK_INDEX} \\
    --master-addr ${VC_WORKER_HOSTS_0} \\
    --install-deps
"""

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import yaml
import torch
import torch.distributed as dist
import torch.nn.functional as F

from model import build_stage
from activation_cache import ActivationCache
from pp_engine import PPEngine
from recovery_protocol import (
    plan_recovery, execute_recovery_sync, execute_recovery_async,
)
from checkpoint_v2 import save_checkpoint, load_checkpoint_no_dist
from coordinator import FileCoordinator, make_coordinator
from mem_tracker import MemoryTracker


# ── 合成数据 ──────────────────────────────────────────────────────────────────

def make_batch(step: int, total_size: int, seq_len: int,
               vocab_size: int, base_seed: int = 42):
    """
    根据 step 生成确定性的合成数据批次。
    同一 step + 同一 seed → 所有 rank 得到相同数据，保证 rank 0 / K-1 对齐。
    返回 (input_ids, targets)，shape=(total_size, seq_len)，LM 移位 target。
    """
    gen = torch.Generator()
    gen.manual_seed(base_seed + step * 104729)   # 大素数避免规律碰撞
    ids = torch.randint(0, vocab_size, (total_size, seq_len + 1), generator=gen)
    return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()


def get_step_data(step, rank, K, M, mb_size, seq_len, vocab_size, seed):
    """返回 (input_ids_or_None, targets_or_None)。
    Ring 拓扑：rank 0 同时拿 input_ids 和 targets（loss 在 rank 0 算）；
    其他 rank 两者均为 None。"""
    if rank != 0:
        return None, None
    ids, tgts = make_batch(step, M * mb_size, seq_len, vocab_size, seed)
    return ids, tgts


# ── 工具 ─────────────────────────────────────────────────────────────────────

def cfg_path(v: str) -> str:
    return os.path.expanduser(os.path.expandvars(v))


def _reinit_pg_filestore(rank: int, world_size: int, coord_dir: str,
                          trial_uid: str, nccl_timeout_sec: int) -> None:
    """
    用 dist.FileStore 重建 NCCL 进程组(需要共享文件系统)。

    FileStore 基于共享文件系统(/workspace/...),不需要任何 TCP 端口。

    关键:每个 trial 使用唯一的文件名(trial_uid = "k{k}_t{trial}"),
    避免不同 trial 间 FileStore 文件残留导致的死锁。
    不需要手动删文件,新路径天然干净。
    """
    store_path = os.path.join(coord_dir, f"recovery_store_{trial_uid}")

    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(
        backend="nccl",
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=nccl_timeout_sec),
        # 显式指定 device_id 避免 NCCL 猜测 GPU 导致的潜在 hang
        device_id=torch.device(f"cuda:{torch.cuda.current_device()}"),
    )


def _reinit_pg_tcpstore(rank: int, world_size: int,
                         master_addr: str, master_port: int,
                         nccl_timeout_sec: int) -> None:
    """
    用 dist.TCPStore 重建 NCCL 进程组(不依赖共享文件系统)。

    每个 trial 用一个新端口,避免上次 trial 残留的 TCPStore 跟新 trial 冲突。
    由调用方负责传入唯一端口(例如 base_port + trial_index)。
    """
    store = dist.TCPStore(
        host_name=master_addr,
        port=master_port,
        world_size=world_size,
        is_master=(rank == 0),
        timeout=timedelta(seconds=nccl_timeout_sec),
        wait_for_workers=True,
    )
    dist.init_process_group(
        backend="nccl",
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=nccl_timeout_sec),
        device_id=torch.device(f"cuda:{torch.cuda.current_device()}"),
    )


# ── 真实断连 / 重连（Phase B）────────────────────────────────────────────────

def do_real_preemption(rank, target_rank, K, engine, optimizer,
                       inject_step, ckpt_dir, coord_dir, coord,
                       master_addr, recovery_port,
                       recovery_nccl_timeout_sec,
                       trial_uid: str = "trial",
                       reinit_kind: str = "filestore") -> dict:
    """
    6 步真实 PG 摧毁 + 重建：

      B1. dist.barrier()                全组最后同步
      B2. target 写 "crashed" 信号      其他节点轮询检测 → detection_sec
      B3. 各自 destroy_process_group()  → pg_destroy_sec
      B4. FileBarrier "pg_destroyed"    等全组完成
      B5. target 从磁盘加载 ckpt+acache → ckpt_load_sec
      B6. FileBarrier "ready_reinit"
          dist.FileStore → dist.init_process_group()  → comm_rebuild_sec（无需 TCP 端口）
    """
    sfx = str(inject_step)

    # B0: target 提前写出 crashed 信号(在任何 collective 之前)
    # 防止 B1 的 dist.barrier() 因 retained graph / NCCL 余量挂起,
    # 导致其他 rank 在 wait_signal('crashed_*') 上 60s 超时退出。
    # B0 后 t_b1 仍然记录 barrier 完成时点用来算其他 rank 的 detection_sec。
    if rank == target_rank:
        coord.signal(f"crashed_{sfx}", str(rank))

    # B1
    dist.barrier()
    t_b1 = time.monotonic()

    if rank == target_rank:
        # B2: (信号已在 B0 写出,这里跳过)
        # B3
        t0 = time.monotonic()
        dist.destroy_process_group()
        pg_destroy_sec = time.monotonic() - t0

        engine.cache.clear_all()
        engine.cache._release_graph()

        # B4
        coord.barrier(f"pg_destroyed_{sfx}", rank)

        # B5
        t0 = time.monotonic()
        load_checkpoint_no_dist(
            engine.stage, optimizer,
            step=inject_step, ckpt_dir=ckpt_dir, rank=rank,
            activation_cache=engine.cache,
            device=f"cuda:{torch.cuda.current_device()}",
        )
        ckpt_load_sec = time.monotonic() - t0

        # B6:重建 PG。两种实现:
        #   - filestore: 共享 /workspace,无需 TCP 端口
        #   - tcpstore : TCPStore,不依赖共享 FS,每个 trial 唯一端口
        coord.barrier(f"ready_reinit_{sfx}", rank)
        t0 = time.monotonic()
        if reinit_kind == "tcpstore":
            _reinit_pg_tcpstore(rank, K, master_addr, recovery_port,
                                recovery_nccl_timeout_sec)
        else:
            _reinit_pg_filestore(rank, K, coord_dir, trial_uid,
                                 recovery_nccl_timeout_sec)
        comm_rebuild_sec = time.monotonic() - t0
        detection_sec = 0.0

    else:
        # B2: 等 crashed 信号(180s,给 GPT-2 XL 量级 destroy+ckpt 留余量)
        coord.wait_signal(f"crashed_{sfx}", timeout=180)
        detection_sec = time.monotonic() - t_b1

        # B3
        t0 = time.monotonic()
        dist.destroy_process_group()
        pg_destroy_sec = time.monotonic() - t0

        # B4
        coord.barrier(f"pg_destroyed_{sfx}", rank)

        ckpt_load_sec = 0.0

        # B6
        coord.barrier(f"ready_reinit_{sfx}", rank)
        t0 = time.monotonic()
        if reinit_kind == "tcpstore":
            _reinit_pg_tcpstore(rank, K, master_addr, recovery_port,
                                recovery_nccl_timeout_sec)
        else:
            _reinit_pg_filestore(rank, K, coord_dir, trial_uid,
                                 recovery_nccl_timeout_sec)
        comm_rebuild_sec = time.monotonic() - t0

    # 验证新 PG
    dist.barrier()

    if rank == 0:
        coord.cleanup_barrier(f"pg_destroyed_{sfx}")
        coord.cleanup_barrier(f"ready_reinit_{sfx}")
        coord.clear_signal(f"crashed_{sfx}")

    return {
        "detection_sec":    detection_sec,
        "pg_destroy_sec":   pg_destroy_sec,
        "ckpt_load_sec":    ckpt_load_sec,
        "comm_rebuild_sec": comm_rebuild_sec,
        "relaunch_sec":     pg_destroy_sec + ckpt_load_sec + comm_rebuild_sec,
    }


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      required=True)
    ap.add_argument("--k",           type=int, required=True,
                    help="被抢占的 stage（1-indexed，即 rank k-1 被抢占）")
    ap.add_argument("--trial",       type=int, required=True)
    ap.add_argument("--master-addr", required=True,
                    help="rank 0 的 IP（= VC_WORKER_HOSTS_0），Phase B 恢复 PG 用")
    ap.add_argument("--master-port", type=int, default=29500)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    inj = cfg["injection"]
    K   = cfg["pp"]["num_stages"]

    if not (1 <= args.k <= K):
        raise ValueError(f"--k 必须在 [1, {K}] 内，got {args.k}")

    # ── 初始化训练 PG（torchrun 已设置 RANK / WORLD_SIZE / LOCAL_RANK）────────
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=inj["nccl_timeout_sec"]),
    )
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)

    target_rank           = args.k - 1
    retain_graph_interval = int(inj.get("retain_graph_interval", 10))
    inject_step           = inj["inject_at_step"]
    M       = cfg["pp"]["num_microbatches"]
    mb_size = cfg["pp"]["micro_batch_size"]
    seq_len = cfg["model"]["max_seq_len"]
    V       = cfg["model"]["vocab_size"]
    H       = cfg["model"]["hidden_dim"]
    seed    = cfg["train"]["seed"]
    ckpt_dir = cfg_path(cfg["checkpoint"]["dir"])

    if retain_graph_interval > 0 and inject_step % retain_graph_interval != 0:
        if rank == 0:
            print(f"[WARN] inject_step={inject_step} % retain_graph_interval="
                  f"{retain_graph_interval} != 0 → 图不会被保留，退化为重做 forward")

    torch.manual_seed(seed + rank)

    # ── 构建 stage / optimizer / engine（从随机初始化，无需任何外部文件）──────
    stage     = build_stage(rank, K, cfg["model"]).cuda()
    optimizer = torch.optim.AdamW(
        stage.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    engine = PPEngine(
        stage, K, M, mb_size, seq_len, H, V,
        cache_max_steps=inj["activation_cache_max_steps"],
        retain_graph_interval=retain_graph_interval,
    )

    coord = make_coordinator(
        inj,
        rank=rank, world_size=K,
        coord_dir=cfg_path(inj["coord_dir"]),
        master_addr=args.master_addr,
    )

    # ── GPU 峰值显存追踪：覆盖整个 trial (Phase A .. Phase D) ────────────────
    # 起点选在 model+optimizer 已构建、warmup 开始前:排除模块级 CUDA context
    # 之类的一次性开销,只统计训练/恢复流程本身触达的峰值。每 rank 各测各的,
    # Phase E 通过 all_gather_object 汇总到 rank 0 写进 JSON。
    mem_tracker = MemoryTracker(device=engine.device)
    mem_tracker.start()

    # ── Phase A：从 step 1 开始 warmup，累积 activation cache ─────────────────
    # 每步都记 wall-clock + loss(rank 0 有值,其他 rank 是 0),供后续外推
    # 稳态 per-step 时长。torch.cuda.synchronize() 在 barrier 前调用,
    # 避免把 kernel 排队开销算进空转时间。
    if rank == 0:
        print(f"[k={args.k} t={args.trial}] Phase A: "
              f"warmup step 1 → {inject_step - 1}  (合成数据，无外部文件)")

    warmup_step_times: list = []      # [{"step":s,"sec":t,"loss":l}]
    t_phase_a_start = time.monotonic()
    for step in range(1, inject_step):
        b, t = get_step_data(step, rank, K, M, mb_size, seq_len, V, seed)
        t0 = time.monotonic()
        loss_v = engine.step(b, t, optimizer, step_id=step)
        torch.cuda.synchronize()
        dist.barrier()
        step_sec = time.monotonic() - t0
        warmup_step_times.append({
            "step": step,
            "sec":  round(step_sec, 4),
            "loss": (round(float(loss_v), 6)
                     if rank == 0 and loss_v is not None else None),
        })
        if rank == 0 and step % 10 == 0:
            print(f"[k={args.k} t={args.trial}] warmup step {step}/{inject_step-1} "
                  f"| {step_sec*1000:.0f} ms/step")
    phase_a_sec = time.monotonic() - t_phase_a_start

    # ── Phase A.6：运行 inject_step，填充 cache，保留 autograd 图 ─────────────
    b_inj, t_inj = get_step_data(inject_step, rank, K, M, mb_size, seq_len, V, seed)
    t0 = time.monotonic()
    pre_loss = engine.step(b_inj, t_inj, optimizer, step_id=inject_step)
    torch.cuda.synchronize()
    dist.barrier()
    phase_a6_sec = time.monotonic() - t0

    # 广播 pre_loss（Ring 拓扑：只有 rank 0 有值）
    pre_t = torch.tensor([float(pre_loss)], device=f"cuda:{local_rank}",
                          dtype=torch.float32)
    dist.broadcast(pre_t, src=0)
    pre_preemption_loss = float(pre_t[0])

    if rank == 0:
        print(f"[k={args.k} t={args.trial}] Phase A.6 done | "
              f"inject_step={inject_step} | "
              f"graph_in_mem={engine.cache.has_graph(inject_step)} | "
              f"pre_loss={pre_preemption_loss:.4f}")

    # ── Phase A.7：model + activation cache 落盘 ──────────────────────────────
    #   "计算图结果磁盘持久化"：
    #     · {ckpt_dir}/step_{inject_step:06d}_rank{r}.pt       model/optimizer/RNG
    #     · {ckpt_dir}/step_{inject_step:06d}_rank{r}_acache.pt detached activation tensors
    #   target 在 Phase B.5 从磁盘恢复，无需重做任何 forward；
    #   upstream helper 若意外崩溃（级联故障）也可从 _acache.pt 恢复。
    if rank == 0:
        print(f"[k={args.k} t={args.trial}] Phase A.7: "
              f"save ckpt + acache → {ckpt_dir}")

    # 计时 checkpoint 保存(每 rank 各测各的:rank 0 多存 head+embed,
    # 其他 rank 只存 blocks。save_checkpoint 内部有 dist.barrier,所以
    # 报告值是"所有 rank 都完成写盘"的时点,per-rank 差异由 all_gather
    # 汇总。activation cache 部分也算在内。)
    t0 = time.monotonic()
    save_checkpoint(stage, optimizer,
                    step=inject_step, ckpt_dir=ckpt_dir,
                    activation_cache=engine.cache,
                    keep_last=cfg["checkpoint"].get("keep_last", 3))
    ckpt_save_sec = time.monotonic() - t0

    # ── Phase B：真实断连 + 重建 PG ──────────────────────────────────────────
    if rank == 0:
        print(f"[k={args.k} t={args.trial}] Phase B: "
              f"real preemption of rank {target_rank}")

    # 根据 coordinator 类型选 PG 重建方式;
    # 同时给 TCPStore 重建用一个 trial-unique 端口,避免上次 trial 残留端口被复用
    coord_kind = str(inj.get("coordinator", "file")).lower()
    if coord_kind == "tcp":
        reinit_kind = "tcpstore"
        # 每个 trial 一个新端口:base + 12*k + 3*trial(+1 是 master rendezvous,跳过)
        reinit_port_base = int(inj.get("reinit_tcp_port_base", 29600))
        recovery_port = reinit_port_base + args.k * 13 + args.trial
    else:
        reinit_kind = "filestore"
        recovery_port = inj.get("recovery_port", args.master_port + 1)

    pt = do_real_preemption(
        rank=rank, target_rank=target_rank, K=K,
        engine=engine, optimizer=optimizer,
        inject_step=inject_step,
        ckpt_dir=ckpt_dir,
        coord_dir=cfg_path(inj["coord_dir"]),
        coord=coord,
        master_addr=args.master_addr,
        recovery_port=recovery_port,
        recovery_nccl_timeout_sec=inj.get("recovery_nccl_timeout_sec",
                                           inj["nccl_timeout_sec"]),
        trial_uid=f"k{args.k}_t{args.trial}",
        reinit_kind=reinit_kind,
    )

    if rank == 0:
        print(f"[k={args.k} t={args.trial}] Phase B done | "
              f"detect={pt['detection_sec']:.3f}s | "
              f"destroy={pt['pg_destroy_sec']:.3f}s | "
              f"ckpt={pt['ckpt_load_sec']:.3f}s | "
              f"reinit={pt['comm_rebuild_sec']:.3f}s | "
              f"relaunch={pt['relaunch_sec']:.3f}s")

    # ── Phase C：差异化恢复（新 PG 上执行）───────────────────────────────────
    plan = plan_recovery(target_rank=target_rank, num_stages=K)

    use_async = bool(inj.get("async_pipeline", False))
    recovery_fn = execute_recovery_async if use_async else execute_recovery_sync

    if rank == 0:
        print(f"[k={args.k} t={args.trial}] Phase C: {recovery_fn.__name__} | "
              f"graph_in_mem={engine.cache.has_graph(inject_step)}")
    rt = recovery_fn(
        plan, engine,
        ckpt_step=inject_step, resume_step=inject_step,
        batch_input_ids=b_inj, batch_targets=t_inj,
        optimizer=optimizer,
    )

    if rank == 0:
        print(f"[k={args.k} t={args.trial}] Phase C done | "
              f"resend={rt['activation_resend_sec']:.3f}s | "
              f"compute={rt['recovery_compute_sec']:.3f}s | "
              f"graph_used={rt['used_retained_graph']} | "
              f"async={rt.get('used_async_pipeline', False)} | "
              f"overlap={rt.get('overlap_sec', 0.0):.3f}s")

    # ── Phase D：继续训练 catch_up_steps 步，收集 loss + 每步 wall-clock ─────
    # 每步计时 → 可以看出恢复后是否有一次性开销(如 NCCL 重新预热、graph 释放
    # 不干净等),或是否稳定回到 Phase A 的 per-step 时长。
    catch_up_steps = inj.get("catch_up_steps", 5)
    if rank == 0:
        print(f"[k={args.k} t={args.trial}] Phase D: {catch_up_steps} catch-up steps")

    loss_traj = []
    catchup_step_times: list = []
    t_phase_d_start = time.monotonic()
    for step in range(inject_step + 1, inject_step + 1 + catch_up_steps):
        b, t = get_step_data(step, rank, K, M, mb_size, seq_len, V, seed)
        t0 = time.monotonic()
        loss = engine.step(b, t, optimizer, step_id=step)
        torch.cuda.synchronize()
        dist.barrier()
        step_sec = time.monotonic() - t0
        catchup_step_times.append({
            "step": step,
            "sec":  round(step_sec, 4),
            "loss": (round(float(loss), 6)
                     if rank == 0 and loss is not None else None),
        })
        if rank == 0:
            loss_traj.append({"step": step, "loss": round(float(loss), 6)})
    phase_d_sec = time.monotonic() - t_phase_d_start

    # ── 停显存追踪 + 跨 rank 汇总（写 JSON 之前) ─────────────────────────────
    # 每 rank 停自己的 tracker,拿到本地峰值 (bytes),
    # 然后 all_gather_object 把 4 份汇到所有 rank,rank 0 再挑最大值/整表写 JSON。
    # all_gather_object 是 collective,必须所有 rank 都参与,故这段在 rank 判断外。
    mem_local = mem_tracker.stop()
    mem_by_rank_list: list = [None] * K
    dist.all_gather_object(mem_by_rank_list, {"rank": rank, **mem_local})
    # {rank: peak_bytes},按 rank 升序,方便 JSON 里直接对照
    mem_by_rank = {int(entry["rank"]): int(entry["peak_bytes"])
                   for entry in mem_by_rank_list}
    peak_max_bytes    = max(mem_by_rank.values())
    peak_max_rank     = max(mem_by_rank, key=mem_by_rank.get)

    # ── 每 rank 的 Phase B/C 时序汇总 ────────────────────────────────────────
    # 每个 rank 都测了自己的 detection / pg_destroy / ckpt_load / comm_rebuild,
    # Phase C 的 resend/compute 也是 per-rank(recovery_protocol 里各 rank 都记了)。
    # 通过 all_gather_object 把 4 份汇总,rank 0 写出:
    #   per_rank_breakdown : 每个 rank 独立时序(含真实 ckpt_load)
    #   wall_clock_breakdown: 保留原语义 = target_rank 视角
    #                          (这样 analyze 脚本读到的是 target 上真实测量的 ckpt_load
    #                           而不是 rank 0 作为 helper 时的 0)
    #   phase_max_breakdown : max-over-ranks 视角(端到端 wall-clock 更贴切)
    local_breakdown = {
        "rank":                  rank,
        "role":                  ("PREEMPTED" if rank == target_rank
                                   else ("UPSTREAM_HELPER" if rank < target_rank
                                         else "DOWNSTREAM_VICTIM")),
        # Phase A/A.6/A.7/D wall-clock (per rank)
        "phase_a_sec":           phase_a_sec,
        "phase_a6_sec":          phase_a6_sec,
        "phase_a7_ckpt_save_sec": ckpt_save_sec,
        "phase_d_sec":           phase_d_sec,
        # Phase B (per rank, from do_real_preemption)
        "detection_sec":         pt["detection_sec"],
        "pg_destroy_sec":        pt["pg_destroy_sec"],
        "ckpt_load_sec":         pt["ckpt_load_sec"],
        "comm_rebuild_sec":      pt["comm_rebuild_sec"],
        "relaunch_sec":          pt["relaunch_sec"],
        # Phase C (per rank, from recovery_fn)
        "activation_resend_sec": rt["activation_resend_sec"],
        "recovery_compute_sec":  rt["recovery_compute_sec"],
        "overlap_sec":           rt.get("overlap_sec", 0.0),
    }
    breakdowns_list: list = [None] * K
    dist.all_gather_object(breakdowns_list, local_breakdown)
    # 按 rank 排序,方便下游读
    per_rank_breakdown = {
        int(b["rank"]): {k: v for k, v in b.items() if k != "rank"}
        for b in breakdowns_list
    }

    # ── Per-step timings 也需要汇总(warmup + catchup 每步) ──────────────────
    # Phase A/D 各 rank 每步的 wall-clock 都测了(带 cuda.synchronize + barrier),
    # 由于每步末尾有 barrier,各 rank 每步的 step_sec 高度接近(差异 <1ms),
    # 但 loss 只有 rank 0 有值,所以直接 broadcast rank 0 的记录当权威版本
    # (per-rank 差异归入 Phase A/D 的 max view)。
    warmup_wire: list = [None] * K
    catchup_wire: list = [None] * K
    dist.all_gather_object(warmup_wire, warmup_step_times)
    dist.all_gather_object(catchup_wire, catchup_step_times)
    per_rank_step_times = {
        "warmup":  {r: warmup_wire[r]  for r in range(K)},
        "catchup": {r: catchup_wire[r] for r in range(K)},
    }

    # ── Phase E：写 JSON（Ring 拓扑：rank 0 持有 loss）─────────────────────────
    if rank == 0:
        rec_loss  = rt.get("final_loss")
        tolerance = inj.get("catch_up_tolerance", 0.05)
        loss_gap  = (abs(float(rec_loss) - pre_preemption_loss)
                     if rec_loss is not None else None)

        converge_step = None
        for entry in loss_traj:
            if abs(entry["loss"] - pre_preemption_loss) <= tolerance:
                converge_step = entry["step"]
                break

        # ── 三份视角的 timing 汇总 ──────────────────────────────────────────
        # 1) target_view : 被抢占那个 rank 上的实测值(含真实 ckpt_load,
        #                   是 pipeline 恢复真正的关键路径)
        # 2) max_view    : 每一项取跨 rank 的最大值(端到端 wall-clock 上界)
        # 3) per_rank    : 完整的 per-rank 明细,方便 debug / 后续分析
        def _view(source: dict, aggregator=None):
            """从 per_rank_breakdown 抽一份视图。
            source: rank -> {key: val}
            aggregator: None => 直接返回 source[target_rank]
                        callable => 对每 key 做跨 rank 聚合"""
            keys = ["detection_sec", "pg_destroy_sec", "ckpt_load_sec",
                    "comm_rebuild_sec", "activation_resend_sec",
                    "recovery_compute_sec", "overlap_sec",
                    "phase_a_sec", "phase_a6_sec",
                    "phase_a7_ckpt_save_sec", "phase_d_sec"]
            if aggregator is None:
                d = source[target_rank]
                out = {k: d[k] for k in keys}
            else:
                out = {k: aggregator([source[r][k] for r in source])
                       for k in keys}
            out["relaunch_sec"] = (out["pg_destroy_sec"] + out["ckpt_load_sec"]
                                    + out["comm_rebuild_sec"])
            # total_recovery_sec:
            #   detection 只在非-target rank 上有值(=从 barrier 到收到 crashed 信号的等待),
            #   端到端 wall-clock ≈ max(target_relaunch, other_detection+other_relaunch)
            #                    + Phase C (resend + compute - overlap)
            # 这里两份视图分别取 target / max,total 一致按当前视图内数值汇总。
            out["total_recovery_sec"] = (
                out["detection_sec"] + out["relaunch_sec"]
                + out["activation_resend_sec"] + out["recovery_compute_sec"]
                - out["overlap_sec"]
            )
            return out

        target_view = _view(per_rank_breakdown)
        max_view    = _view(per_rank_breakdown, aggregator=max)

        result = {
            "k":           args.k,
            "trial":       args.trial,
            "K_total":     K,
            "target_rank": target_rank,
            "inject_step": inject_step,
            "retain_graph_interval": retain_graph_interval,
            "used_retained_graph":    rt["used_retained_graph"],
            "used_async_pipeline":    rt.get("used_async_pipeline", False),

            # 【向后兼容】默认 wall_clock_breakdown = target_rank 的实测,
            # 这样老 analyze 脚本读到的 ckpt_load 就不再是 rank 0 helper 的 0,
            # 而是 target 上真实的 load 耗时(含 rank-0 head+embed 特殊情况)。
            "wall_clock_breakdown": target_view,

            # 新增:max-over-ranks 视图(端到端 wall-clock 更贴切)
            "wall_clock_breakdown_max": max_view,

            # 新增:每 rank 完整明细,含角色标签
            "per_rank_breakdown": per_rank_breakdown,

            # ★ 新增:整个 trial 从头到尾的 Phase 级 wall-clock 汇总
            # 报告 max-over-ranks(端到端所有 rank 都完成才算这个 Phase 结束)
            # 各 phase 有 barrier 拉齐,per-rank 差异一般 <10ms
            "phase_timings": {
                "phase_a_warmup_total_sec":    max(per_rank_breakdown[r]["phase_a_sec"]
                                                    for r in per_rank_breakdown),
                "phase_a6_inject_step_sec":    max(per_rank_breakdown[r]["phase_a6_sec"]
                                                    for r in per_rank_breakdown),
                "phase_a7_ckpt_save_sec_max":  max(per_rank_breakdown[r]["phase_a7_ckpt_save_sec"]
                                                    for r in per_rank_breakdown),
                "phase_a7_ckpt_save_sec_target": per_rank_breakdown[target_rank]["phase_a7_ckpt_save_sec"],
                "phase_b_relaunch_sec_target":   per_rank_breakdown[target_rank]["relaunch_sec"],
                "phase_b_relaunch_sec_max":      max(per_rank_breakdown[r]["relaunch_sec"]
                                                      for r in per_rank_breakdown),
                "phase_c_recovery_sec_target":   target_view["total_recovery_sec"],
                "phase_c_recovery_sec_max":      max_view["total_recovery_sec"],
                "phase_d_catchup_total_sec":     max(per_rank_breakdown[r]["phase_d_sec"]
                                                     for r in per_rank_breakdown),
                # trial 端到端 wall-clock(A..D 的总时长,近似 pipeline 从零训到恢复完成后 5 步)
                "trial_end_to_end_sec": (
                    max(per_rank_breakdown[r]["phase_a_sec"]  for r in per_rank_breakdown)
                    + max(per_rank_breakdown[r]["phase_a6_sec"] for r in per_rank_breakdown)
                    + max(per_rank_breakdown[r]["phase_a7_ckpt_save_sec"] for r in per_rank_breakdown)
                    + max_view["total_recovery_sec"]
                    + max(per_rank_breakdown[r]["phase_d_sec"]  for r in per_rank_breakdown)
                ),
            },

            # ★ 新增:每步的 wall-clock + loss 时序(rank 0 视角权威)
            # warmup: step 1..inject_step-1  各步耗时 + loss(可画完整 loss 曲线)
            # catchup: inject_step+1..+catch_up_steps  各步耗时 + loss
            "per_step_timings": {
                "warmup":  warmup_step_times,      # rank 0 记录
                "catchup": catchup_step_times,     # rank 0 记录
                # per-rank 数据留在这里,方便 debug 各 rank 有无严重不均
                "per_rank_warmup":  per_rank_step_times["warmup"],
                "per_rank_catchup": per_rank_step_times["catchup"],
            },

            "recovery_breakdown": {
                "stages_resent_activation":  rt["stages_resent_activation"],
                "stages_recomputed_forward": rt["stages_recomputed_forward"],
            },

            # 每 rank 的 GPU 峰值显存 (bytes),覆盖 Phase A..D 整个 trial 窗口。
            # 用 torch.cuda.max_memory_allocated,只算 tensor+autograd graph
            # 实际持有的部分,不含 caching allocator reserve 出来的余量。
            # peak_bytes_max 是四个 rank 的最大值,方便一眼看 "最紧那卡多大"。
            "gpu_memory": {
                "peak_bytes_max":       peak_max_bytes,
                "peak_max_mib":         round(peak_max_bytes / (1024 * 1024), 2),
                "peak_max_rank":        peak_max_rank,
                "peak_bytes_per_rank":  mem_by_rank,
                "peak_mib_per_rank": {r: round(b / (1024 * 1024), 2)
                                      for r, b in mem_by_rank.items()},
            },

            "convergence": {
                "pre_preemption_loss": pre_preemption_loss,
                "recovery_loss":  float(rec_loss) if rec_loss is not None else None,
                "loss_gap":       round(float(loss_gap), 6) if loss_gap is not None else None,
                "converge_step":  converge_step,
                "catch_up_tolerance": tolerance,
                "loss_trajectory":    loss_traj,
            },

            "notes": (
                "Fully self-contained: synthetic data, train from scratch. "
                "Real PG destroy+reinit (tcp://). "
                f"retain_graph_interval={retain_graph_interval}. "
                f"used_retained_graph={rt['used_retained_graph']}. "
                "acache persisted to disk. No external files needed."
            ),
        }

        results_dir = Path(cfg_path(cfg["output"]["results_dir"]))
        results_dir.mkdir(parents=True, exist_ok=True)
        out = results_dir / f"injection_k{args.k}_trial{args.trial}.json"
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"[result] saved {out}")
        # 一行汇总打印,方便日志肉眼扫:哪张卡最紧、多大
        peak_mib_str = ", ".join(f"r{r}={m:.0f}MiB"
                                 for r, m in sorted(
                                     result["gpu_memory"]["peak_mib_per_rank"].items()))
        print(f"[k={args.k} t={args.trial}] GPU peak: "
              f"max={result['gpu_memory']['peak_max_mib']:.0f}MiB "
              f"(rank {peak_max_rank}) | per-rank: {peak_mib_str}")

    try:
        dist.destroy_process_group()
    except Exception:
        pass


if __name__ == "__main__":
    main()
