"""Unit tests for jacobian_fro_norm against an explicit autograd Jacobian.

torch required; skip if unavailable.
"""

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

from Utils.grad_operations import jacobian_fro_norm


def _explicit_fro_norm(f, x):
    """Compute ||J_f(x_i)||_F explicitly via per-output autograd."""
    B = x.size(0)
    y = f(x)  # (B, m)
    m = y.shape[1]
    out = torch.zeros(B)
    for k in range(m):
        grad = torch.autograd.grad(
            y[:, k].sum(), x, retain_graph=True, create_graph=False, only_inputs=True
        )[0]  # (B, d)
        out = out + (grad ** 2).sum(dim=1)
    return torch.sqrt(out + 1e-12)


def test_fro_norm_linear():
    torch.manual_seed(0)
    B, d, m = 8, 5, 3
    f = nn.Linear(d, m)
    x = torch.randn(B, d)
    g = jacobian_fro_norm(f, x, create_graph=False)
    g_ref = _explicit_fro_norm(f, x)
    # For a linear map, J_f is constant = weight matrix W (m x d)
    W = f.weight.detach()
    expected = torch.full((B,), W.norm().item())
    assert g.shape == (B,)
    assert torch.allclose(g, g_ref, atol=1e-4)
    assert torch.allclose(g, expected, atol=1e-4)


def test_fro_norm_mlp_matches_explicit():
    torch.manual_seed(1)
    B, d, hid, m = 6, 4, 16, 2
    f = nn.Sequential(nn.Linear(d, hid), nn.Tanh(), nn.Linear(hid, m))
    x = torch.randn(B, d)
    g = jacobian_fro_norm(f, x, create_graph=False)
    g_ref = _explicit_fro_norm(f, x)
    assert torch.allclose(g, g_ref, atol=1e-4)
