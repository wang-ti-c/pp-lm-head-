"""Tests for the head-at-stage-0 ring layout."""
import torch
from model import build_stage, StageFirst, StageLast, StageMiddle


def _cfg(H=8, V=16, L=2, heads=2, seq=4, dropout=0.0):
    return {
        "hidden_dim": H, "vocab_size": V, "num_layers_per_stage": L,
        "num_heads": heads, "max_seq_len": seq, "dropout": dropout,
    }


def test_stage_first_has_head_and_ln_f():
    s = StageFirst(_cfg())
    assert hasattr(s, "lm_head")
    assert hasattr(s, "ln_f")
    assert hasattr(s, "tok_emb")
    assert hasattr(s, "pos_emb")


def test_stage_first_no_legacy_forward():
    """StageFirst.forward must be removed; only forward_embed / forward_head exist."""
    s = StageFirst(_cfg())
    # StageFirst must NOT override nn.Module.forward — this is the invariant
    # that pp_engine._forward_with_clones_method's monkey-patch cleanup relies on
    # (when the cleanup restores forward, the inherited base-class forward must
    # take over and raise NotImplementedError, not a stale override).
    assert "forward" not in StageFirst.__dict__, \
        "StageFirst must not override nn.Module.forward"
    # Both new entry-point methods must exist:
    assert callable(getattr(s, "forward_embed", None))
    assert callable(getattr(s, "forward_head", None))


def test_stage_first_forward_embed_shape_and_grad():
    cfg = _cfg(H=8, V=16, seq=4)
    s = StageFirst(cfg)
    ids = torch.randint(0, cfg["vocab_size"], (2, cfg["max_seq_len"]))
    h = s.forward_embed(ids)
    assert h.shape == (2, cfg["max_seq_len"], cfg["hidden_dim"])
    # gradient flows back to tok_emb
    h.sum().backward()
    assert s.tok_emb.weight.grad is not None


def test_stage_first_forward_head_shape_and_grad():
    cfg = _cfg(H=8, V=16, seq=4)
    s = StageFirst(cfg)
    hidden = torch.randn(2, cfg["max_seq_len"], cfg["hidden_dim"],
                          requires_grad=True)
    logits = s.forward_head(hidden)
    assert logits.shape == (2, cfg["max_seq_len"], cfg["vocab_size"])
    logits.sum().backward()
    assert hidden.grad is not None
    assert s.lm_head.weight.grad is not None
    assert s.ln_f.weight.grad is not None


def test_stage_last_no_head():
    s = StageLast(_cfg())
    assert not hasattr(s, "lm_head")
    assert not hasattr(s, "ln_f")
    assert hasattr(s, "blocks")


def test_stage_last_forward_preserves_shape():
    cfg = _cfg(H=8, seq=4)
    s = StageLast(cfg)
    x = torch.randn(2, cfg["max_seq_len"], cfg["hidden_dim"])
    y = s(x)
    assert y.shape == x.shape


def test_build_stage_rank0_is_stage_first():
    cfg = _cfg()
    assert isinstance(build_stage(0, 4, cfg), StageFirst)


def test_build_stage_rankK_minus_1_is_stage_last():
    cfg = _cfg()
    assert isinstance(build_stage(3, 4, cfg), StageLast)


def test_build_stage_middle():
    cfg = _cfg()
    assert isinstance(build_stage(1, 4, cfg), StageMiddle)
    assert isinstance(build_stage(2, 4, cfg), StageMiddle)
