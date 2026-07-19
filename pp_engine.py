"""
Pipeline-parallel 训练引擎（Ring 拓扑 + 图优化版）。

Ring 拓扑差异：
  · 新增 4 个 ring-closure 通信原语：_send_hidden_to_head / _recv_hidden_from_tail
                                   _send_grad_to_tail   / _recv_grad_from_head
  · rank 0 既驱动 embed forward（送给 rank 1），又消费来自 rank K-1 的 hidden 跑 head + loss
  · rank K-1 退化为中间 stage 同构：forward 完把 hidden 发回 rank 0，等 grad 回来 backward
  · do_retain_here 仍排除 K-1 —— Ring 拓扑下 K-1 结构同构于中间 stage，
    但按 plan_recovery 定义 K-1 永远不会成为 UPSTREAM_HELPER，
    保留的图无消费者，排除它避免无谓显存

retain_graph 行为：
  · rank 0 的 embed 段：与原版一致，走 functional_call + cloned params 路径
  · rank 0 的 head 段：每 mb backward 后图自动释放，无需保留
  · rank K-1：与中间 stage 同形态，但不参与 retain_graph_interval（无消费者）
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.func import functional_call

from activation_cache import ActivationCache
from pair_groups import ensure_pair_groups, pair_send, pair_recv


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

    # ── 通信原语（链式 + ring 回传） ─────────────────────────────────────────
    #
    # 所有 P2P 都走 pair sub-PG broadcast，不再用 raw dist.send/recv。
    # 原因 (2026-07 fix)：NCCL 2.19+ 对 4-rank PG 上的 P2P 采用 LAZY 2-rank
    # sub-comm 创建，而该创建是父 PG 上的隐式 collective。Ring 拓扑下
    # (0→1, 1→2, 2→3, 3→0) 4 条边在不同 rank 上首次触达顺序不同 → 死锁,
    # 每个 rank 在第一个 forward mb 上 hang 满 timeout。observed on
    # A100 x4, NCCL 2.27.5, 训练 warmup step 1 一步都过不去。
    # 解法: pair_groups 模块预建 4 条 pair sub-PG + warm-up，业务通信
    # 走 broadcast(src=lo, group=pair_group)，等价 send/recv 但零 lazy-init。

    def _recv_act(self):
        buf = torch.empty((self.mb_size, self.seq_len, self.H),
                          device=self.device, dtype=torch.float32)
        pair_recv(buf, src=self.rank - 1)
        return buf

    def _send_act(self, x):
        pair_send(x.contiguous(), dst=self.rank + 1)

    def _recv_grad(self):
        buf = torch.empty((self.mb_size, self.seq_len, self.H),
                          device=self.device, dtype=torch.float32)
        pair_recv(buf, src=self.rank + 1)
        return buf

    def _send_grad(self, g):
        pair_send(g.contiguous(), dst=self.rank - 1)

    def _send_hidden_to_head(self, x):
        """rank K-1 → rank 0：forward 时把最末 hidden 送回 head"""
        pair_send(x.contiguous(), dst=0)

    def _recv_hidden_from_tail(self):
        """rank 0 接收来自 rank K-1 的 hidden（forward 末段）"""
        buf = torch.empty((self.mb_size, self.seq_len, self.H),
                          device=self.device, dtype=torch.float32)
        pair_recv(buf, src=self.K - 1)
        return buf

    def _send_grad_to_tail(self, g):
        """rank 0 → rank K-1：backward 启动时把 grad_hidden 送给 tail"""
        pair_send(g.contiguous(), dst=self.K - 1)

    def _recv_grad_from_head(self):
        """rank K-1 接收来自 rank 0 的 grad_hidden（backward 启动）"""
        buf = torch.empty((self.mb_size, self.seq_len, self.H),
                          device=self.device, dtype=torch.float32)
        pair_recv(buf, src=0)
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
        # 通过 monkey-patch self.stage.forward 让 functional_call 调用指定方法。
        # nn.Module 的 forward 始终在基类定义（默认 raise NotImplementedError），
        # 所以 hasattr(self.stage, "forward") 恒为 True，无需 None 分支兜底。
        method = getattr(self.stage, method_name)
        saved_forward = self.stage.forward
        self.stage.forward = method  # type: ignore[assignment]
        try:
            out = functional_call(self.stage, cloned, (inp,))
        finally:
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
        # Bootstrap pair sub-PGs before ANY P2P. Idempotent — re-fires only
        # if the default PG changed (Phase B rebuild) or on first call.
        # Must be called by ALL ranks in lock-step (collective).
        ensure_pair_groups(self.K, self.device)

        self.current_step = step_id
        do_retain = (self.retain_graph_interval > 0
                     and step_id % self.retain_graph_interval == 0)
        # Ring 拓扑下 K-1 与中间 stage 结构同构，但按 plan_recovery 的定义
        # K-1 永远不会是 UPSTREAM_HELPER（HELPER 只分配给 rank < target_rank，
        # 而 target_rank ≤ K-1），所以 K-1 保留的图永远没有消费者。
        # 排除 K-1，避免无谓的峰值显存占用。
        do_retain_here = do_retain and (self.rank != self.K - 1)

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
