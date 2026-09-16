"""Tests for new corruption modes (mask, dropout)."""
import torch

from Utils.projectors import CorruptionConfig, CorruptionOperator


def test_mask_corrupt_shape_and_frac():
    cfg = CorruptionConfig(mode="mask", mask_frac=0.3)
    Pi = CorruptionOperator(cfg)
    x = torch.ones(32, 10)
    x_t, t = Pi(x)
    assert x_t.shape == x.shape
    assert t is None
    zero_frac = (x_t == 0).float().mean().item()
    assert 0.2 < zero_frac < 0.4, f"expected ~30% zeros, got {zero_frac:.3f}"


def test_dropout_corrupt_shape_and_keep():
    cfg = CorruptionConfig(mode="dropout", drop_p=0.4)
    Pi = CorruptionOperator(cfg)
    x = torch.ones(32, 10)
    x_t, t = Pi(x)
    assert x_t.shape == x.shape
    assert t is None
    zero_frac = (x_t == 0).float().mean().item()
    assert 0.3 < zero_frac < 0.5, f"expected ~40% zeros, got {zero_frac:.3f}"


def test_mask_corrupt_no_change_at_frac_zero():
    cfg = CorruptionConfig(mode="mask", mask_frac=0.0)
    Pi = CorruptionOperator(cfg)
    x = torch.randn(16, 5)
    x_t, _ = Pi(x)
    assert torch.allclose(x, x_t)


def test_dropout_corrupt_no_change_at_drop_zero():
    cfg = CorruptionConfig(mode="dropout", drop_p=0.0)
    Pi = CorruptionOperator(cfg)
    x = torch.randn(16, 5)
    x_t, _ = Pi(x)
    assert torch.allclose(x, x_t)
