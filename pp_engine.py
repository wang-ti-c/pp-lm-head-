"""
Pipeline-parallel 训练引擎（图优化版）。

核心改动：
  当 do_retain_here=True（inject_step 且 rank < K-1）时，
  forward 改用 torch.func.functional_call + 参数副本（clone）运行。

  原理：
    autograd 图引用 cloned 参数（version=0），而非 stage 原始参数。
    optimizer.step() 对原始参数做 inplace 更新，cloned 参数不受影响，
    其 version 始终为 0，与图建立时一致。
    Phase C 调用 g_out.backward() 时 version 检查通过，不再崩溃。

  backward：
    梯度先累积在 cloned 参数的 .grad，再手动 copy 到原始参数的 .grad，
    供 optimizer.step() 正常更新。
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

    # ── 通信原语 ──────────────────────────────────────────────────────────────

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

    # ── 参数副本 forward（图优化专用）────────────────────────────────────────

    def _forward_with_clones(self, inp):
        """
        用 functional_call + 参数副本运行 forward。

        autograd 图引用 cloned_params（version=0），不引用 stage 原始参数。
        optimizer.step() 修改原始参数后，cloned_params version 仍为 0，
        Phase C 的 g_out.backward() 可以安全执行。

        Returns: (out, cloned_params_dict)
        """
        cloned = {n: p.detach().clone().requires_grad_(True)
                  for n, p in self.stage.named_parameters()}
        out = functional_call(self.stage, cloned, (inp,))
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
        # rank K-1 永远不是 UPSTREAM_HELPER，不需要保留图
        do_retain_here = do_retain and (self.rank < self.K - 1)

        optimizer.zero_grad()
        fwd_in:    list[torch.Tensor] = []
        fwd_out:   list[torch.Tensor] = []
        losses:    list[torch.Tensor] = []
        clone_sets: list              = []  # None 或 cloned_params_dict

        # ── Forward ───────────────────────────────────────────────────────────
        for mb in range(self.M):

            if self.rank == 0:
                s   = mb * self.mb_size
                inp = batch_input_ids[s:s + self.mb_size].to(self.device)
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

            elif self.rank == self.K - 1:
                inp = self._recv_act()
                inp.requires_grad_(True)
                fwd_in.append(inp)
                logits = self.stage(inp)
                fwd_out.append(logits)
                clone_sets.append(None)
                s   = mb * self.mb_size
                tgt = batch_targets[s:s + self.mb_size].to(self.device)
                losses.append(
                    F.cross_entropy(logits.view(-1, self.V), tgt.view(-1)) / self.M
                )

            else:
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

        # ── Backward ──────────────────────────────────────────────────────────
        for mb in range(self.M):
            cloned = clone_sets[mb]

            if self.rank == self.K - 1:
                losses[mb].backward()
                self._send_grad(fwd_in[mb].grad)

            elif self.rank == 0:
                g = self._recv_grad()
                fwd_out[mb].backward(g, retain_graph=do_retain_here)
                if cloned is not None:
                    self._transfer_grads(cloned)

            else:
                g = self._recv_grad()
                fwd_out[mb].backward(g, retain_graph=do_retain_here)
                self._send_grad(fwd_in[mb].grad)
                if cloned is not None:
                    self._transfer_grads(cloned)

        optimizer.step()

        if do_retain_here:
            self.last_graph_step = step_id

        return sum(l.item() for l in losses) if self.rank == self.K - 1 else 0.0
