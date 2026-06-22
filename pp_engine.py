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
