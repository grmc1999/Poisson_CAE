"""Unit tests for the localized Poisson estimators (L1 compact-support, L2 kNN).

Key guarantees checked:
  - As R -> inf (L1) or k -> M (L2), the localized estimator recovers the global
    PoissonMCEstimator.
  - The Wendland window has the right endpoint behaviour.
  - Outputs are finite and correctly shaped.
  - The estimators are differentiable w.r.t the landmark/source inputs.
  - Diffusion kernel: no r^{2-d} overflow at high d.
  - Variational (Ritz) estimator: produces finite v, gradv; energy decreases.

torch required; skip if unavailable.
"""

import pytest

torch = pytest.importorskip("torch")

import math

from Utils.geometry_estimators import PoissonMCConfig, PoissonMCEstimator
from Utils.localized_estimators import (
    CompactSupportEstimator,
    CompactSupportConfig,
    KNNEstimator,
    KNNConfig,
    wendland_c2,
)
from Utils.grad_operations import (
    diffusion_reg,
    gradx_diffusion_reg,
    green_reg,
    gradx_green_reg,
)
from Utils.variational_estimator import (
    VariationalEstimator,
    VariationalConfig,
    PotentialHead,
    ritz_energy,
)


@pytest.fixture
def setup2d():
    torch.manual_seed(0)
    B, M, d = 8, 32, 2
    x = torch.randn(B, d)
    x_land = torch.randn(M, d)
    g_land = torch.rand(M) + 0.5
    return x, x_land, g_land


@pytest.fixture
def setup_highd():
    torch.manual_seed(3)
    B, M, d = 8, 24, 50
    x = torch.randn(B, d)
    x_land = torch.randn(M, d)
    g_land = torch.rand(M) + 0.5
    return x, x_land, g_land


def _global(x, x_land, g_land, eps=0.01):
    est = PoissonMCEstimator(PoissonMCConfig(eps=eps, landmarks=x_land.shape[0]))
    return est.forward(x, x_land, g_land)


# ============================================================
# Poisson kernel tests (require kernel_type='poisson')
# ============================================================

# -----------------------------
# Wendland window
# -----------------------------
def test_wendland_endpoints():
    R = 2.0
    assert wendland_c2(torch.tensor([0.0]), R).item() == pytest.approx(1.0)
    assert wendland_c2(torch.tensor([R]), R).item() == pytest.approx(0.0)
    assert wendland_c2(torch.tensor([2.0 * R]), R).item() == pytest.approx(0.0)
    assert abs(wendland_c2(torch.tensor([0.999 * R]), R).item()) < 1e-4


# -----------------------------
# L1 consistency (Poisson)
# -----------------------------
def test_l1_recovers_global_large_radius(setup2d):
    x, x_land, g_land = setup2d
    M = x_land.shape[0]
    est = CompactSupportEstimator(
        CompactSupportConfig(
            eps=0.01, radius=1e6, max_neighbors=M,
            normalize_by_cardinality=False, kernel_type="poisson",
        )
    )
    v_loc, gradv_loc = est(x, x_land, g_land)
    v_glo, gradv_glo = _global(x, x_land, g_land)
    assert v_loc.shape == (x.shape[0],)
    assert gradv_loc.shape == x.shape
    assert torch.allclose(v_loc, v_glo, atol=1e-5)
    assert torch.allclose(gradv_loc, gradv_glo, atol=1e-5)


def test_l1_with_window_matches_dense_masked(setup2d):
    x, x_land, g_land = setup2d
    R = 0.8
    eps = 0.01
    M = x_land.shape[0]
    est = CompactSupportEstimator(
        CompactSupportConfig(
            eps=eps, radius=R, max_neighbors=M,
            normalize_by_cardinality=False, kernel_type="poisson",
        )
    )
    v_loc, gradv_loc = est(x, x_land, g_land)

    G = green_reg(x, x_land, eps=eps)
    dG = gradx_green_reg(x, x_land, eps=eps)
    sq = ((x[:, None, :] - x_land[None, :, :]) ** 2).sum(-1)
    r = torch.sqrt(sq + eps ** 2)
    w = wendland_c2(r, R)
    v_ref = (w * G * g_land[None, :]).sum(1) / M
    gradv_ref = (w[:, :, None] * dG * g_land[None, :, None]).sum(1) / M

    assert torch.allclose(v_loc, v_ref, atol=1e-5)
    assert torch.allclose(gradv_loc, gradv_ref, atol=1e-5)


# -----------------------------
# L2 consistency (Poisson)
# -----------------------------
def test_l2_recovers_global_full_k(setup2d):
    x, x_land, g_land = setup2d
    M = x_land.shape[0]
    est = KNNEstimator(KNNConfig(eps=0.01, k=M, normalize_by_k=True, kernel_type="poisson"))
    v_loc, gradv_loc = est(x, x_land, g_land)
    v_glo, gradv_glo = _global(x, x_land, g_land)
    assert v_loc.shape == (x.shape[0],)
    assert gradv_loc.shape == x.shape
    assert torch.allclose(v_loc, v_glo, atol=1e-5)
    assert torch.allclose(gradv_loc, gradv_glo, atol=1e-5)


def test_l2_single_neighbor_recovers_knn_pair(setup2d):
    x, x_land, g_land = setup2d
    eps = 0.01
    est = KNNEstimator(KNNConfig(eps=eps, k=1, normalize_by_k=True, kernel_type="poisson"))
    v_loc, gradv_loc = est(x, x_land, g_land)

    sq = ((x[:, None, :] - x_land[None, :, :]) ** 2).sum(-1)
    idx = sq.argmin(dim=1)
    near = x_land[idx]
    for b in range(x.shape[0]):
        y = near[b : b + 1]
        G = green_reg(x[b : b + 1], y, eps=eps)[0, 0]
        dG = gradx_green_reg(x[b : b + 1], y, eps=eps)[0, 0]
        assert v_loc[b].item() == pytest.approx(G.item() * g_land[idx[b]].item())
        assert torch.allclose(gradv_loc[b], dG * g_land[idx[b]].item(), atol=1e-6)


# -----------------------------
# Differentiation / finiteness (Poisson)
# -----------------------------
def test_localized_differentiable_wrt_landmarks(setup2d):
    x, x_land, g_land = setup2d
    xl = x_land.clone().requires_grad_(True)
    for est in (
        CompactSupportEstimator(CompactSupportConfig(eps=0.01, radius=0.8, max_neighbors=16, kernel_type="poisson")),
        KNNEstimator(KNNConfig(eps=0.01, k=8, kernel_type="poisson")),
    ):
        _, gradv = est(x, xl, g_land)
        loss = (gradv ** 2).sum()
        assert loss.requires_grad
        loss.backward()
        assert xl.grad is not None
        assert torch.isfinite(xl.grad).all()
        xl.grad = None


def test_high_dim_shapes_and_finite_poisson(setup_highd):
    x, x_land, g_land = setup_highd
    for est in (
        CompactSupportEstimator(CompactSupportConfig(eps=0.01, radius=2.0, max_neighbors=8, kernel_type="poisson")),
        KNNEstimator(KNNConfig(eps=0.01, k=8, kernel_type="poisson")),
    ):
        v, gradv = est(x, x_land, g_land)
        assert v.shape == (x.shape[0],)
        assert gradv.shape == x.shape
        assert torch.isfinite(v).all()
        assert torch.isfinite(gradv).all()


# ============================================================
# Diffusion kernel tests
# ============================================================

def test_diffusion_kernel_values():
    x = torch.randn(4, 2)
    y = torch.randn(6, 2)
    G = diffusion_reg(x, y, t=1.0)
    assert G.shape == (4, 6)
    assert torch.all(G > 0)
    assert torch.all(G <= 1.0)


def test_diffusion_kernel_finite_at_high_d():
    torch.manual_seed(3)
    B, M, d = 8, 24, 50
    x = torch.randn(B, d)
    y = torch.randn(M, d)
    G = diffusion_reg(x, y, t=1.0)
    dG = gradx_diffusion_reg(x, y, t=1.0)
    assert G.shape == (B, M)
    assert dG.shape == (B, M, d)
    assert torch.isfinite(G).all()
    assert torch.isfinite(dG).all()


def test_diffusion_kernel_finite_at_mnist_dim():
    """The Poisson kernel dies at d=784; diffusion must survive with proper t.

    Key insight: in R^d, typical inter-point distance is sqrt(d), so r^2 ~ d.
    The Gaussian exp(-r^2/4t) requires t ~ O(d) to avoid underflow.
    """
    torch.manual_seed(42)
    B, M, d = 8, 16, 784
    t = d / 4.0          # scale t with d so that exp(-r^2/4t) ≈ exp(-1) for typical distances
    x = torch.randn(B, d)
    y = torch.randn(M, d)
    G = diffusion_reg(x, y, t=t)
    dG = gradx_diffusion_reg(x, y, t=t)
    assert torch.isfinite(G).all()
    assert torch.isfinite(dG).all()
    # gradient magnitudes are nonzero (not all dead)
    assert dG.abs().max().item() > 1e-30


def test_diffusion_gradient_analytic_matches_autograd():
    """∇_x G_t(x, y) matches autograd on the full (B, M, d) output."""
    torch.manual_seed(7)
    x = torch.randn(3, 4, requires_grad=True)
    y = torch.randn(5, 4)
    t = 0.5

    G = diffusion_reg(x, y, t=t)
    dG_analytic = gradx_diffusion_reg(x, y, t=t)

    # Autograd: sum G w.r.t x, then check grad matches analytic gradient
    dG_autograd = torch.autograd.grad(G.sum(), x, create_graph=False)[0]

    # dG_autograd = sum over M of ∇_x G_t(x_i, y_j) = sum_j dG_analytic[i, :, :]
    dG_analytic_summed = dG_analytic.sum(dim=1)  # (B, d)
    assert torch.allclose(dG_autograd, dG_analytic_summed, atol=1e-6)


def test_diffusion_localized_estimator_shaped_and_finite(setup_highd):
    x, x_land, g_land = setup_highd
    for est in (
        CompactSupportEstimator(CompactSupportConfig(
            eps=0.01, radius=2.0, max_neighbors=8, kernel_type="diffusion")),
        KNNEstimator(KNNConfig(eps=0.01, k=8, kernel_type="diffusion")),
    ):
        v, gradv = est(x, x_land, g_land)
        assert v.shape == (x.shape[0],)
        assert gradv.shape == x.shape
        assert torch.isfinite(v).all()
        assert torch.isfinite(gradv).all()


def test_diffusion_knn_consistency_vs_dense(setup2d):
    """L2 with k=M using diffusion kernel should match the dense diffusion sum."""
    x, x_land, g_land = setup2d
    M = x_land.shape[0]
    eps = 0.01
    t = 0.8

    knn = KNNEstimator(KNNConfig(eps=eps, k=M, normalize_by_k=True, kernel_type="diffusion", t=t))
    v_loc, gradv_loc = knn(x, x_land, g_land)

    G = diffusion_reg(x, x_land, t=t)
    dG = gradx_diffusion_reg(x, x_land, t=t)
    v_ref = (G * g_land[None, :]).sum(1) / M
    gradv_ref = (dG * g_land[None, :, None]).sum(1) / M

    assert torch.allclose(v_loc, v_ref, atol=1e-5)
    assert torch.allclose(gradv_loc, gradv_ref, atol=1e-5)


# ============================================================
# Variational (Ritz) estimator tests
# ============================================================

def test_variational_finite_and_shaped(setup2d):
    x, x_land, g_land = setup2d
    cfg = VariationalConfig(inner_steps=3, inner_lr=0.01, mu=0.0, lam_d=1.0)
    est = VariationalEstimator(cfg, d=x.shape[1])
    v, gradv = est(x, x_land, g_land)
    assert v.shape == (x.shape[0],)
    assert gradv.shape == x.shape
    assert torch.isfinite(v).all()
    assert torch.isfinite(gradv).all()


def test_variational_highd_finite(setup_highd):
    x, x_land, g_land = setup_highd
    cfg = VariationalConfig(inner_steps=3, inner_lr=0.01, mu=0.0, lam_d=1.0)
    est = VariationalEstimator(cfg, d=x.shape[1])
    v, gradv = est(x, x_land, g_land)
    assert torch.isfinite(v).all()
    assert torch.isfinite(gradv).all()


def test_variational_energy_decreases():
    """The Ritz energy should decrease over K inner steps (on average)."""
    torch.manual_seed(0)
    d = 4
    M = 64
    x_land = torch.randn(M, d)
    s = torch.rand(M) + 0.1
    v = PotentialHead(d, h=64, layers=2)
    opt = torch.optim.SGD(v.parameters(), lr=0.01)

    energies = []
    for _ in range(20):
        opt.zero_grad()
        e = ritz_energy(v, x_land, s, x_bd=x_land, mu=0.0, lam_d=1.0)
        e.backward()
        opt.step()
        energies.append(e.item())

    assert energies[-1] < energies[5]


def test_variational_energy_decreases_mu_screened():
    """Ritz energy decreases with mu > 0."""
    torch.manual_seed(1)
    d = 4
    M = 64
    x_land = torch.randn(M, d)
    s = torch.rand(M) + 0.1
    v = PotentialHead(d, h=64, layers=2)
    opt = torch.optim.SGD(v.parameters(), lr=0.01)

    energies = []
    for _ in range(20):
        opt.zero_grad()
        e = ritz_energy(v, x_land, s, x_bd=x_land, mu=0.1, lam_d=1.0)
        e.backward()
        opt.step()
        energies.append(e.item())

    assert energies[-1] < energies[5]
