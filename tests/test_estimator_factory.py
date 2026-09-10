"""Tests for Utils.estimator_factory (Phase 2 estimator switchboard)."""

import pytest

import torch

from Utils.config import EstimatorConfig
from Utils.estimator_factory import build_estimator, default_diffusion_t


def _call(est, B=8, M=16, d=4):
    est.eval()
    xq = torch.randn(B, d)
    xl = torch.randn(M, d)
    g = torch.rand(M).abs() + 1e-3
    v, gv = est(xq, xl, g)
    return v.detach(), gv.detach()


def test_default_diffusion_t_rule():
    assert default_diffusion_t(784) == pytest.approx(196.0)
    assert default_diffusion_t(2) == pytest.approx(0.5)


def test_build_global():
    est = build_estimator(EstimatorConfig(scheme="global"), d=2)
    v, gv = _call(est, d=2)
    assert v.shape == (8,)
    assert gv.shape == (8, 2)


def test_build_compact_diffusion():
    est = build_estimator(
        EstimatorConfig(scheme="compact", kernel_type="diffusion", t=1.0,
                        radius=2.0, max_neighbors=16, normalize=True),
        d=4,
    )
    v, gv = _call(est)
    assert torch.isfinite(v).all()
    assert torch.isfinite(gv).all()
    assert gv.shape == (8, 4)


def test_build_knn_diffusion_t_fallback():
    # t <= 0 triggers the d/4 high-d fallback
    est = build_estimator(EstimatorConfig(scheme="knn", kernel_type="diffusion",
                                          t=0.0, k=8), d=32)
    assert est.cfg.t == pytest.approx(8.0)
    v, gv = _call(est, d=32)
    assert torch.isfinite(v).all()
    assert torch.isfinite(gv).all()


def test_build_variational():
    est = build_estimator(EstimatorConfig(scheme="variational", inner_steps=2), d=4)
    v, gv = _call(est)
    assert v.shape == (8,)
    assert gv.shape == (8, 4)


def test_build_unknown_scheme_raises():
    with pytest.raises(ValueError):
        build_estimator(EstimatorConfig(scheme="nope"), d=2)