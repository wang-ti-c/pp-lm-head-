"""
Per-rank GPU peak-memory tracker.

Usage
-----
    tracker = MemoryTracker(device=engine.device)
    tracker.start()          # Phase A start (resets PyTorch's peak counter)
    ... run training / recovery ...
    stats = tracker.stop()   # Phase D end
    # stats = {"peak_bytes": int}

Peak is read from PyTorch's built-in max_memory_allocated() counter —
accurate, zero-cost (updated by CUDA caching allocator as tensors are
alloc'd/freed), no background sampling needed.

Falls back to a no-op (peak_bytes=0) on CPU / when CUDA is unavailable so
tests that construct engine.device = torch.device("cpu") don't blow up.
"""
import torch


class MemoryTracker:
    def __init__(self, device):
        self.device = device
        # CPU / no-CUDA fallback: everything becomes a no-op.
        # device is either torch.device or a legacy int index; treat both.
        self._enabled = torch.cuda.is_available() and (
            not isinstance(device, torch.device) or device.type == "cuda"
        )

    def start(self) -> None:
        """Reset PyTorch's internal peak counter so max_memory_allocated()
        reports the peak over THIS window only. Idempotent."""
        if not self._enabled:
            return
        torch.cuda.reset_peak_memory_stats(self.device)

    def stop(self) -> dict:
        """Return {'peak_bytes': int}. On CPU/no-CUDA returns 0."""
        if not self._enabled:
            return {"peak_bytes": 0}
        return {"peak_bytes": int(torch.cuda.max_memory_allocated(self.device))}
