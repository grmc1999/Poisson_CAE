"""Tests for new corruption modes (mask, dropout)."""
import torch
import torch.nn.functional as F

from Utils.projectors import CorruptionConfig, CorruptionOperator, affine_sample_2d


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


def test_affine_sample_2d_matches_grid_sample():
    torch.manual_seed(0)
    B, C, H, W = 4, 1, 28, 28
    img = torch.randn(B, C, H, W, dtype=torch.float64)
    theta = torch.zeros(B, 2, 3, dtype=torch.float64)
    for b in range(B):
        ang = torch.rand(()).item() * 0.6 - 0.3
        scale = 1.0 + torch.rand(()).item() * 0.3 - 0.15
        theta[b, 0, 0] = torch.cos(torch.tensor(ang)) / scale
        theta[b, 0, 1] = -torch.sin(torch.tensor(ang)) / scale
        theta[b, 1, 0] = torch.sin(torch.tensor(ang)) / scale
        theta[b, 1, 1] = torch.cos(torch.tensor(ang)) / scale

    ref = F.grid_sample(img, F.affine_grid(theta, img.shape, align_corners=False),
                        align_corners=False)
    out = affine_sample_2d(img, theta)
    assert torch.allclose(out, ref, atol=1e-10)

    img1 = img.clone().requires_grad_(True)
    img2 = img.clone().requires_grad_(True)
    g1 = torch.autograd.grad(affine_sample_2d(img1, theta).sum(), img1)[0]
    ref2 = F.grid_sample(img2, F.affine_grid(theta, img.shape, align_corners=False),
                         align_corners=False)
    g2 = torch.autograd.grad(ref2.sum(), img2)[0]
    assert torch.allclose(g1, g2, atol=1e-10)


def test_affine_sample_2d_supports_double_backward():
    # F.grid_sample cannot be double-differentiated; the bilevel estimator needs
    # this, so the manual sampler must pass gradgradcheck.
    torch.manual_seed(0)
    B, H, W = 3, 28, 28
    img = torch.randn(B, 1, H, W, dtype=torch.float64)
    theta = torch.zeros(B, 2, 3, dtype=torch.float64)
    theta[:, 0, 0] = 0.9
    theta[:, 1, 1] = 0.9
    inp = img.clone().requires_grad_(True)
    assert torch.autograd.gradgradcheck(
        lambda z: (affine_sample_2d(z, theta) ** 2).mean(), inp
    )

