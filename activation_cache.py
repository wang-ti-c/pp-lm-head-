"""
跨 step 的 activation 缓存。

每个 rank 维护两份 cache：
  _data       : detached + clone，用于恢复阶段的 forward（upstream resend）
  _graph_data : 不 detach，保留 autograd 图，用于 upstream helper 直接 backward
                （跳过重做 forward，节省计算时间）

_graph_data 每次只保留最近一个 retained step，防止 OOM。
retain_graph_interval = 0 时完全禁用图缓存，退化为原始行为。
"""
from collections import OrderedDict
import torch


class ActivationCache:
    def __init__(self, max_steps: int = 200, retain_graph_interval: int = 0):
        self.max_steps             = max_steps
        self.retain_graph_interval = retain_graph_interval
        self._data: "OrderedDict[int, dict]" = OrderedDict()
        self._graph_data: dict      = {}
        self._graph_step: int | None = None

    # ── Detached cache ────────────────────────────────────────────────────────

    def save(self, step, mb, inp, out):
        if self.max_steps == 0:
            return
        if step not in self._data:
            self._data[step] = {}
            while len(self._data) > self.max_steps:
                self._data.popitem(last=False)
        self._data[step][mb] = (inp.detach().clone(), out.detach().clone())

    def get(self, step, mb):
        return self._data[step][mb]

    def has_step(self, step):
        return step in self._data and len(self._data[step]) > 0

    def clear_before(self, ckpt_step):
        for s in [s for s in self._data if s < ckpt_step]:
            del self._data[s]

    def clear_all(self):
        self._data.clear()

    # ── Graph cache ───────────────────────────────────────────────────────────

    def save_with_graph(self, step, mb, inp, out):
        self.save(step, mb, inp, out)
        if self._graph_step is not None and self._graph_step != step:
            self._release_graph()
        self._graph_step = step
        if step not in self._graph_data:
            self._graph_data[step] = {}
        self._graph_data[step][mb] = (inp, out)

    def has_graph(self, step):
        return (self.retain_graph_interval > 0
                and self._graph_step == step
                and step in self._graph_data
                and len(self._graph_data[step]) > 0)

    def get_graph(self, step, mb):
        return self._graph_data[step][mb]

    def _release_graph(self):
        self._graph_data.clear()
        self._graph_step = None

    def release_graph_explicitly(self):
        self._release_graph()

    def summary(self):
        steps = sorted(self._data)
        d = f"steps=[{steps[0]}..{steps[-1]}]({len(steps)})" if steps else "empty"
        g = f"graph_step={self._graph_step}" if self._graph_step is not None else "no_graph"
        return f"ActivationCache({d}, {g})"
