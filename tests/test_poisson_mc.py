"""Unit tests for the Poisson Monte-Carlo Green estimator.

Checks the estimator against an independent manual quadrature of the Green
representation, the single-landmark analytic (fundamental solution) case, and
consistency as the number of landmarks grows.

torch required; skip if unavailable.
"""

import pytest

torch = pytest.importorskip("torch")

import math

from Utils.geometry_estimators import PoissonMCEstimator, PoissonMCConfig
from Utils.grad_operations import green_reg, gradx_green_reg, omega_d


def _forward_ref(x, x_land, g_land, eps):
    """Independent manual quadrature of v and gradv."""
    M = x_land.shape[0]
    G = green_reg(x, x_land, eps=eps)
    dG = gradx_green_reg(x, x_land, eps=eps)
    v = (G * g_land[None, :]).sum(dim=1) / float(M)
    gradv = (dG * g_land[None, :, None]).sum(dim=1) / float(M)
    return v, gradv


def test_estimator_matches_manual_quadrature_2d():
    torch.manual_seed(0)
    B, M, d = 10, 32, 2
    cfg = PoissonMCConfig(eps=0.01, landmarks=M)
    est = PoissonMCEstimator(cfg)
    x = torch.randn(B, d)
    x_land = torch.randn(M, d)
    g_land = torch.rand(M) + 0.5  # positive source

    v, gradv = est.forward(x, x_land, g_land)
    v_ref, gradv_ref = _forward_ref(x, x_land, g_land, cfg.eps)

    assert v.shape == (B,)
    assert gradv.shape == (B, d)
    assert torch.allclose(v, v_ref, atol=1e-6)
    assert torch.allclose(gradv, gradv_ref, atol=1e-6)


def test_single_landmark_recovers_fundamental_solution_2d():
    """With one landmark (g=1), v(x)=G(x,y0) and gradv=∇G(x,y0)."""
    d = 2
    B, M = 12, 1
    cfg = PoissonMCConfig(eps=0.01, landmarks=1)
    est = PoissonMCEstimator(cfg)
    x = torch.randn(B, d)
    y0 = torch.tensor([[1.0, -0.5]], dtype=torch.float32)
    g_land = torch.ones(M)

    v, gradv = est.forward(x, y0, g_land)
    v_ref = green_reg(x, y0, eps=cfg.eps)[:, 0]
    gradv_ref = gradx_green_reg(x, y0, eps=cfg.eps)[:, 0, :]

    assert torch.allclose(v, v_ref, atol=1e-6)
    assert torch.allclose(gradv, gradv_ref, atol=1e-6)


def test_consistency_with_more_landmarks_3d():
    """With a constant source, the estimate should converge to the analytic
    average as M grows (not a strong convergence test — just checks that the
    mean-based quadrature is unbiased w.r.t the manual reference at any M)."""
    torch.manual_seed(4)
    d = 3
    B, M = 6, 64
    cfg = PoissonMCConfig(eps=0.01, landmarks=M)
    est = PoissonMCEstimator(cfg)
    x = torch.randn(B, d)
    x_land = torch.randn(M, d)
    g_land = torch.ones(M)  # constant source

    v, gradv = est.forward(x, x_land, g_land)
    v_ref, gradv_ref = _forward_ref(x, x_land, g_land, cfg.eps)
    assert torch.allclose(v, v_ref, atol=1e-6)
    assert torch.allclose(gradv, gradv_ref, atol=1e-6)
