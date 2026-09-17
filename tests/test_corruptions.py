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


def test_rotation_point_preserves_norm_and_identity_at_zero():
    x = torch.randn(32, 2)
    x_t, t = CorruptionOperator(CorruptionConfig(mode="rotation", rotation_max_deg=45.0))(x)
    assert x_t.shape == x.shape and t is None
    assert torch.allclose(x_t.norm(dim=1), x.norm(dim=1), atol=1e-5)

    x_id, _ = CorruptionOperator(CorruptionConfig(mode="rotation", rotation_max_deg=0.0))(x)
    assert torch.allclose(x, x_id)


def test_rotation_image_shape_and_identity_at_zero():
    x = torch.rand(16, 28 * 28)
    cfg = CorruptionConfig(mode="rotation", rotation_max_deg=30.0, image_side=28)
    Pi = CorruptionOperator(cfg)
    x_t, _ = Pi(x)
    assert x_t.shape == x.shape
    assert not torch.allclose(x, x_t)  # nonzero angle scrambles pixels

    x_id, _ = CorruptionOperator(
        CorruptionConfig(mode="rotation", rotation_max_deg=0.0, image_side=28)
    )(x)
    assert torch.allclose(x, x_id, atol=1e-5)


def test_zoom_scales_and_identity_at_zero():
    x = torch.ones(16, 4)
    x_t, t = CorruptionOperator(CorruptionConfig(mode="zoom", zoom_std=0.5))(x)
    assert x_t.shape == x.shape and t is None
    assert (x_t != 1.0).any()
    assert ((x_t > 0).all() and (x_t < 2).all())

    x_id, _ = CorruptionOperator(CorruptionConfig(mode="zoom", zoom_std=0.0))(x)
    assert torch.allclose(x, x_id)


def test_zoom_image_shape():
    x = torch.rand(8, 28 * 28)
    x_t, _ = CorruptionOperator(
        CorruptionConfig(mode="zoom", zoom_std=0.2, image_side=28)
    )(x)
    assert x_t.shape == x.shape
