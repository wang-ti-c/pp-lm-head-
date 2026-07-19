"""Ring-layout invariant tests (CPU, no distributed).

These tests lock the contracts that callers (PPEngine, recovery_protocol,
run_injection_v2) rely on:
  · StageFirst has both forward_embed and forward_head
  · StageLast has neither ln_f nor lm_head
  · plan_recovery target_rank=0 produces the expected role assignment
"""
import os
import sys
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import build_stage, StageFirst, StageLast
from recovery_protocol import plan_recovery, RecoveryRole, execute_recovery_async


def _cfg():
    return {
        "hidden_dim": 8, "vocab_size": 16, "num_layers_per_stage": 2,
        "num_heads": 2, "max_seq_len": 4, "dropout": 0.0,
    }


def test_stage_first_dual_methods_exist():
    s = build_stage(0, 4, _cfg())
    assert isinstance(s, StageFirst)
    assert callable(s.forward_embed)
    assert callable(s.forward_head)


def test_stage_first_has_head_params():
    s = build_stage(0, 4, _cfg())
    names = {n for n, _ in s.named_parameters()}
    # Must include both embed-side and head-side params
    assert any("tok_emb" in n for n in names)
    assert any("lm_head" in n for n in names)
    assert any("ln_f" in n for n in names)


def test_stage_last_no_head_no_ln_f():
    s = build_stage(3, 4, _cfg())
    assert isinstance(s, StageLast)
    assert not hasattr(s, "lm_head"), "StageLast must not own lm_head in ring layout"
    assert not hasattr(s, "ln_f"), "StageLast must not own ln_f in ring layout"


def test_stage_last_param_names_pure_blocks():
    s = build_stage(3, 4, _cfg())
    names = {n for n, _ in s.named_parameters()}
    assert not any("lm_head" in n for n in names)
    assert not any("ln_f" in n for n in names)
    assert any("blocks" in n for n in names)


def test_plan_recovery_target_0():
    """target_rank=0 is the special case: no UPSTREAM_HELPER exists."""
    plan = plan_recovery(target_rank=0, num_stages=4)
    assert plan[0] == RecoveryRole.PREEMPTED
    for r in (1, 2, 3):
        assert plan[r] == RecoveryRole.DOWNSTREAM_VICTIM
    helpers = [r for r, ro in plan.items() if ro == RecoveryRole.UPSTREAM_HELPER]
    assert helpers == [], "target_rank=0 must produce zero helpers"


def test_plan_recovery_target_K_minus_1():
    """target_rank=K-1 maximizes helpers (rank 0/1/2 all helpers)."""
    plan = plan_recovery(target_rank=3, num_stages=4)
    for r in (0, 1, 2):
        assert plan[r] == RecoveryRole.UPSTREAM_HELPER
    assert plan[3] == RecoveryRole.PREEMPTED


def test_async_recovery_uses_ring_head_on_stage0():
    """Async recovery must preserve the ring invariant: rank 0 owns head/loss."""
    src = inspect.getsource(execute_recovery_async)
    assert "_send_hidden_to_head" in src
    assert "_recv_hidden_from_tail" in src
    assert "forward_head" in src
    assert "final_loss = (sum(l.item() for l in losses)\n                  if rank == 0 else None)" in src
    assert "logits = engine.stage(inp)" not in src
