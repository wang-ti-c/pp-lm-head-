"""
增强型 checkpoint：model + optimizer + RNG + activation cache 磁盘持久化。

文件命名：
  {dir}/step_{step:06d}_rank{rank}.pt          — model / optimizer / RNG
  {dir}/step_{step:06d}_rank{rank}_acache.pt   — activation cache detached 快照

save_checkpoint  : 需要 dist 进程组（训练阶段调用）
load_checkpoint  : 需要 dist 进程组（起始加载用，本版不使用）
load_checkpoint_no_dist : 不需要 dist（PG 摧毁后，target rank 重启时调用）
"""
import os
import glob
import torch
from collections import OrderedDict
from pathlib import Path


def save_checkpoint(stage_module, optimizer, step: int, ckpt_dir: str,
                    activation_cache=None, keep_last: int = 3) -> None:
    """
    保存 model + optimizer + RNG（以及可选的 activation cache 快照）。
    需要 dist 进程组（用于 barrier 和 rank 查询）。
    """
    import torch.distributed as dist
    rank = dist.get_rank()
    os.makedirs(ckpt_dir, exist_ok=True)

    # model / optimizer / RNG
    _atomic_save({
        "step":      step,
        "rank":      rank,
        "model":     stage_module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng":  torch.cuda.get_rng_state(),
    }, os.path.join(ckpt_dir, f"step_{step:06d}_rank{rank}.pt"))

    # activation cache（detached tensors → CPU）
    if activation_cache is not None:
        _atomic_save(
            _serialize_cache(activation_cache),
            os.path.join(ckpt_dir, f"step_{step:06d}_rank{rank}_acache.pt"),
        )

    dist.barrier()
    if rank == 0:
        _cleanup(ckpt_dir, keep_last)
    dist.barrier()


def load_checkpoint_no_dist(stage_module, optimizer, step: int,
                             ckpt_dir: str, rank: int,
                             activation_cache=None,
                             device: str = None) -> int:
    """
    在 dist PG 被销毁后加载 checkpoint（rank 由调用方传入）。
    """
    if device is None:
        device = f"cuda:{torch.cuda.current_device()}"

    path = os.path.join(ckpt_dir, f"step_{step:06d}_rank{rank}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location=device, weights_only=False)
    stage_module.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(_coerce_u8(state["torch_rng"]))
    torch.cuda.set_rng_state(_coerce_u8(state["cuda_rng"]))

    if activation_cache is not None:
        apath = os.path.join(ckpt_dir, f"step_{step:06d}_rank{rank}_acache.pt")
        if os.path.exists(apath):
            _restore_cache(activation_cache,
                           torch.load(apath, map_location="cpu", weights_only=False),
                           device)
    return state["step"]


# ── 内部工具 ──────────────────────────────────────────────────────────────────

def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _coerce_u8(t: torch.Tensor) -> torch.Tensor:
    return t.cpu().to(torch.uint8)


def _serialize_cache(cache) -> dict:
    data = {}
    for step, mbs in cache._data.items():
        data[step] = {mb: (inp.detach().cpu(), out.detach().cpu())
                      for mb, (inp, out) in mbs.items()}
    return {"data": data,
            "max_steps": cache.max_steps,
            "retain_graph_interval": cache.retain_graph_interval}


def _restore_cache(cache, saved: dict, device: str):
    cache._data.clear()
    new_data = OrderedDict()
    for step, mbs in sorted(saved["data"].items()):
        new_data[step] = {mb: (inp.to(device), out.to(device))
                          for mb, (inp, out) in mbs.items()}
        while len(new_data) > cache.max_steps:
            new_data.popitem(last=False)
    cache._data = new_data


def _cleanup(ckpt_dir: str, keep_last: int):
    files = glob.glob(os.path.join(ckpt_dir, "step_*_rank0.pt"))
    if len(files) <= keep_last:
        return
    files.sort()
    for step in [int(Path(f).name.split("_")[1]) for f in files[:-keep_last]]:
        for pat in (f"step_{step:06d}_rank*.pt",
                    f"step_{step:06d}_rank*_acache.pt"):
            for f in glob.glob(os.path.join(ckpt_dir, pat)):
                try:
                    os.remove(f)
                except OSError:
                    pass
