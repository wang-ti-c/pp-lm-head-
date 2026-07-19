"""
Ring-topology pair sub-PG helpers, shared by training and recovery.

Why this module exists
----------------------
NCCL 2.19+ implements P2P (dist.send/dist.recv) on a multi-rank PG via
LAZY creation of 2-rank sub-communicators. That "lazy create" is an
implicit collective on the parent PG. On a 4-rank ring topology
(0→1, 1→2, 2→3, 3→0) different ranks touch different pairs first,
so each rank ends up calling the collective in a different order →
deadlock at the very first forward mb. Every worker sits in NCCL SEND
until watchdog kills it 600 s later (observed).

Fix: pre-create every pipeline-edge sub-PG up front (a collective the
whole default PG participates in, in fixed edge order), warm it up
with one broadcast, then send/recv via `dist.broadcast(tensor, src=lo,
group=pair_group)` instead of raw `dist.send/recv`. Broadcast on a
2-rank sub-PG is a strict send/recv with no lazy-init and no deadlock.

The 4 edges we build for a K-rank ring:
    (0,1), (1,2), (2,3), ..., (K-2, K-1)   ← linear neighbours
    (0, K-1)                                ← ring closure

`ensure_pair_groups()` is idempotent-per-default-PG:
if the default PG changes (Phase B tears it down and rebuilds), the
cache is invalidated and everything is rebuilt.

`recovery_protocol.py` used to define these locally; both training and
recovery now import from here so they share one warmed-up cache.
"""
import torch
import torch.distributed as dist


# Cache of dist.new_group results, keyed by (lo, hi) pair.
_PAIR_GROUPS: dict = {}
# Fingerprint of the default PG that populated the cache; invalidated when
# Phase B re-inits (id() changes).
_PAIR_GROUPS_PG_ID: int | None = None


def ensure_pair_groups(K: int, device) -> None:
    """
    Idempotently create dist.new_group for every ring edge.

    MUST be called by ALL ranks in the default PG in lock-step — new_group
    is a collective; ranks that aren't members of a new group still have
    to participate in the same call sequence.

    Detects default-PG re-init (Phase B) by comparing id() of the default
    group; on change, wipes the cache and rebuilds. Warm-up broadcasts
    force NCCL to actually finalize the sub-comm now, so business calls
    later pay zero setup cost.
    """
    global _PAIR_GROUPS, _PAIR_GROUPS_PG_ID

    default_pg = dist.distributed_c10d._get_default_group()
    cur_id = id(default_pg)
    if _PAIR_GROUPS_PG_ID != cur_id:
        _PAIR_GROUPS.clear()
        _PAIR_GROUPS_PG_ID = cur_id

    rank = dist.get_rank()
    one  = torch.zeros(1, device=device, dtype=torch.float32)
    zero = torch.zeros(1, device=device, dtype=torch.float32)

    # Fixed edge order so every rank's new_group calls line up.
    edges = [(src, src + 1) for src in range(K - 1)]
    if K > 1:
        edges.append((0, K - 1))

    for src, dst in edges:
        key = (src, dst)
        if key not in _PAIR_GROUPS:
            grp = dist.new_group(ranks=[src, dst])
            _PAIR_GROUPS[key] = grp
            # Warm-up: only the two members participate.
            # Two broadcasts (one from each side) fully round-trip the
            # sub-comm so the first business call is instant.
            if rank == src:
                dist.broadcast(one,  src=src, group=grp)
                dist.broadcast(zero, src=dst, group=grp)
            elif rank == dst:
                dist.broadcast(one,  src=src, group=grp)
                dist.broadcast(zero, src=dst, group=grp)
            # non-members: idle (must NOT call broadcast on a group they
            # aren't in — that would error).


def pair_group(a: int, b: int):
    """Return the (a,b) pair PG (order-independent)."""
    lo, hi = min(a, b), max(a, b)
    return _PAIR_GROUPS[(lo, hi)]


def pair_send(tensor: torch.Tensor, dst: int, async_op: bool = False):
    """
    P2P send via 2-rank pair sub-PG broadcast.

    Equivalent to dist.send(tensor, dst) but immune to the NCCL 2.19+
    lazy-sub-comm deadlock. Returns a Work handle iff async_op=True.
    """
    grp = pair_group(dist.get_rank(), dst)
    return dist.broadcast(tensor.contiguous(), src=dist.get_rank(),
                          group=grp, async_op=async_op)


def pair_recv(tensor: torch.Tensor, src: int) -> None:
    """
    P2P recv via 2-rank pair sub-PG broadcast. Blocks until the peer
    issues its matching send.
    """
    grp = pair_group(src, dist.get_rank())
    dist.broadcast(tensor, src=src, group=grp)
