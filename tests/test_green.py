"""Unit tests for green_reg / gradx_green_reg.

These validate the regularized Green's function of the Laplacian against
analytic references and against each other via finite differences.

torch is required; skip if unavailable (e.g. on a bare local Python).
"""

import pytest

torch = pytest.importorskip("torch")

import math

from Utils.grad_operations import green_reg, gradx_green_reg, omega_d


def _fund_2d(x, y, eps):
    """Reference: 2D regularized fundamental solution -(1/2pi) ln(sqrt(r^2+eps^2))."""
    r2 = ((x - y) ** 2).sum(-1)
    r = torch.sqrt(r2 + eps ** 2)
    return -(1.0 / (2 * math.pi)) * torch.log(r)


def _fund_3d(x, y, eps):
    """Reference: 3D regularized fundamental solution c / r^(d-2)."""
    d = 3
    c = 1.0 / ((d - 2.0) * omega_d(d))
    r2 = ((x - y) ** 2).sum(-1)
    r = torch.sqrt(r2 + eps ** 2)
    return c * r ** (2.0 - d)


@pytest.mark.parametrize("d,ref", [(2, _fund_2d), (3, _fund_3d)])
def test_green_reg_matches_analytic(d, ref):
    torch.manual_seed(0)
    B, M = 8, 5
    x = torch.randn(B, d)
    y = torch.randn(M, d)
    eps = 0.01

    G = green_reg(x, y, eps=eps)
    G_ref = ref(x[:, None, :], y[None, :, :], eps)
    assert G.shape == (B, M)
    assert torch.allclose(G, G_ref, atol=1e-6), (G, G_ref)


def test_green_grad_matches_autograd_2d():
    torch.manual_seed(1)
    d = 2
    B, M = 4, 3
    x = torch.randn(B, d)
    y = torch.randn(M, d)
    eps = 0.01

    xg = x.clone().requires_grad_(True)
    G = green_reg(xg, y, eps)  # (B,M)
    # exact gradient of each element w.r.t its own x row
    B, M = G.shape
    dG_ref = torch.zeros(B, M, d)
    for b in range(B):
        for m in range(M):
            (grad,) = torch.autograd.grad(G[b, m], xg, retain_graph=True, only_inputs=True)
            dG_ref[b, m] = grad[b]

    dG = gradx_green_reg(x, y, eps)
    assert torch.allclose(dG, dG_ref, atol=1e-5)


def test_green_grad_matches_autograd_3d():
    torch.manual_seed(2)
    d = 3
    B, M = 4, 3
    x = torch.randn(B, d)
    y = torch.randn(M, d)
    eps = 0.01

    xg = x.clone().requires_grad_(True)
    G = green_reg(xg, y, eps)
    Bm, Mm = G.shape
    dG_ref = torch.zeros(Bm, Mm, d)
    for b in range(Bm):
        for m in range(Mm):
            (grad,) = torch.autograd.grad(G[b, m], xg, retain_graph=True, only_inputs=True)
            dG_ref[b, m] = grad[b]

    dG = gradx_green_reg(x, y, eps)
    assert torch.allclose(dG, dG_ref, atol=1e-5)


def test_green_symmetry():
    """G(x,y) should be symmetric between query and landmark sets."""
    torch.manual_seed(3)
    d = 2
    A = torch.randn(5, d)
    B = torch.randn(4, d)
    eps = 0.01
    G_AB = green_reg(A, B, eps)
    G_BA = green_reg(B, A, eps).t()
    assert torch.allclose(G_AB, G_BA, atol=1e-6)
