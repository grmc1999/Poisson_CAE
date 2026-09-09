"""Localized Poisson estimators.

The global `PoissonMCEstimator` (Utils/geometry_estimators.py) evaluates, for every
query point, a dense (B, M) Green matrix and a (B, M, d) gradient, coupling each
query to *every* landmark. Memory is O(B * M * d) and every kernel evaluation is
global -- infeasible at high input dimension (e.g. MNIST, d=784) with large M.

This module provides two localized alternatives that keep *exactly* the same
`forward(x_query, x_land, g_land) -> (v_hat, gradv_hat)` interface so they plug into
`Poisson_reg.Estimate_field_grads` and hence into BC_loss / D_loss unchanged.

Both estimators approximate the Poisson potential

    v(x) = ∫_Ω G(x, y) s(y) dy ,        s(y) = ||J_f(y)||_F^2

via the regularized Laplacian fundamental solution G_eps(x,y), but restrict the
quadrature to a local neighbourhood of each query. The critical implementation
point is that *neighbours are gathered first*: the kernel (Green function and its
gradient) is only evaluated on the O(B * n) gathered pairs, giving O(B * n * d)
memory instead of O(B * M * d).

  L1 - CompactSupportEstimator
        v_hat(x_i) = (1/M_i) Σ_{j: r_ij <= R} w(r_ij/R) G_eps(x_i, y_j) s(y_j)
      Uses the compactly-supported Wendland C^2 window w to keep only landmarks
      in the Euclidean ball N_R(x_i) = { y : r <= R }. w is the Green's function
      of a *screened* Poisson problem, giving the truncation a physical meaning
      (see PLAN.md). A configurable bound n (<= number of landmarks) caps the
      gathered block, so memory is O(B * n * d) while the window still enforces
      the exact ball (window = 0 beyond R).

  L2 - KNNEstimator
        v_hat(x_i) = (1/N_i) Σ_{j ∈ N_k(x_i)} G_eps(x_i, y_j) s(y_j)
      Uses the k nearest landmarks N_k(x_i) per query (exact top-k).

Consistency: as R -> inf (L1; window -> 1 everywhere) or k -> M / n -> M (L2),
the localized quadrature recovers the global estimator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .grad_operations import green_reg, gradx_green_reg, diffusion_reg, gradx_diffusion_reg

# -----------------------------
# Kernel dispatch
# -----------------------------
def _compute_kernel(
    diff: torch.Tensor,
    sqdist: torch.Tensor,
    d: int,
    kernel_type: str = "diffusion",
    eps: float = 1e-2,
    t: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute kernel G and gradient dG for a batch of (diff, sqdist).

    Args:
      diff:       (B, n, d) x_query - x_neigh.
      sqdist:     (B, n) squared Euclidean distance.
      d:          input dimension.
      kernel_type: 'poisson' | 'diffusion'.
      eps:        regularization for the Poisson kernel.
      t:          diffusion scale for the heat kernel.

    Returns:
      G:  (B, n) kernel values.
      dG: (B, n, d) gradients w.r.t x_query.
    """
    r = torch.sqrt(sqdist + eps ** 2)                   # (B, n)

    if kernel_type == "diffusion":
        G = torch.exp(-sqdist / (4.0 * max(t, 1e-12)))                   # (B, n)
        dG = -(diff / (2.0 * max(t, 1e-12))) * G[:, :, None]            # (B, n, d)
        return G, dG

    # Poisson Green (original)
    if d == 2:
        G = -(1.0 / (2.0 * math.pi)) * torch.log(r)
        dG = -(1.0 / (2.0 * math.pi)) * diff / (sqdist[:, :, None] + eps ** 2)
    elif d >= 3:
        c = 1.0 / ((d - 2.0) * omega_d(d))
        G = c * (r ** (2.0 - d))
        dG = c * (2.0 - d) * diff * (r[:, :, None] ** (-d))
    else:
        raise ValueError("d must be >= 2")
    return G, dG


# -----------------------------
# Wendland C^2 window
# -----------------------------
def wendland_c2(r: torch.Tensor, R: float) -> torch.Tensor:
    """Compact-support Wendland C^2 window w(r) with radius R.

    w(r) = (1 - r/R)^4_+ * (1 + 4 r/R)    for 0 <= r <= R,  0 otherwise.

    w is radial, C^2-continuous, w(0)=1, w(R)=0, w'(R)=w''(R)=0, compactly
    supported on the ball of radius R. Differentiable w.r.t r (hence w.r.t
    x_query / x_land), including across the R boundary where w -> 0 smoothly.
    """
    u = r / max(float(R), 1e-12)
    inside = (u < 1.0).to(r.dtype)
    w = (1.0 - u) ** 4 * (1.0 + 4.0 * u)
    return inside * w


# -----------------------------
# Shared neighbor gather
# -----------------------------
def _gather_neighbors(
    x_query: torch.Tensor,
    x_land: torch.Tensor,
    g_land: torch.Tensor,
    n: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather the `n` nearest landmarks (and their source values) of each query.

    Args:
      x_query: (B, d) query points.
      x_land:  (M, d) landmark points.
      g_land:  (M,) source values s(y_j) attached to the landmarks.
      n:       number of neighbours to gather per query (n <= M).

    Returns:
      neigh:    (B, n, d) the n nearest landmark coords per query.
      sqdist:   (B, n) squared Euclidean distance query<->neighbour.
      g_gath:   (B, n) the source value corresponding to each gathered landmark.
    """
    B, d = x_query.shape
    M = x_land.shape[0]
    n = min(n, M)

    # Dense (B, M) for the top-k search. This is O(B*M) in B and M only, NOT d;
    # the d-dimensional kernel evaluation below is restricted to the gathered
    # block (B, n, d), which is where the dominant memory/FLOP win lives.
    sq_full = ((x_query[:, None, :] - x_land[None, :, :]) ** 2).sum(dim=2)  # (B, M)

    _, idx = sq_full.topk(n, dim=1, largest=False, sorted=True)  # (B, n) indices
    # idx -> (B, n, 1) scatter into landmark coordinates
    neigh = torch.gather(
        x_land[None, :, :].expand(B, M, d), 1, idx.unsqueeze(-1).expand(B, n, d)
    )  # (B, n, d)
    sqdist = torch.gather(sq_full, 1, idx)          # (B, n)
    g_gath = torch.gather(g_land[None, :].expand(B, M), 1, idx)  # (B, n)

    return neigh, sqdist, g_gath


# -----------------------------
# L1 - Compact-support / screened kernel
# -----------------------------
@dataclass
class CompactSupportConfig:
    eps: float = 1e-2
    radius: float = 1.0     # R: support radius of the Wendland window
    # Number of nearest landmarks gathered per query. Caps memory at O(B*n*d).
    # Set to 0 to use all landmarks (n = M) -> window still enforces the ball.
    max_neighbors: int = 64
    # Normalize by the number of active (window>0) landmarks per query (True) or
    # by the raw quadrature denominator (False, unnormalized global-style mean).
    normalize_by_cardinality: bool = True
    kernel_type: str = "diffusion"  # 'poisson' | 'diffusion'
    t: float = 1.0                  # diffusion scale (ignored if kernel_type='poisson')


class CompactSupportEstimator(nn.Module):
    """L1: compact-support (Wendland C^2) localized Green quadrature."""

    def __init__(self, cfg: CompactSupportConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, x_query, x_land, g_land) -> Tuple[torch.Tensor, torch.Tensor]:
        B, d = x_query.shape
        M = x_land.shape[0]
        n = min(self.cfg.max_neighbors, M) if self.cfg.max_neighbors > 0 else M

        neigh, sqdist, g_gath = _gather_neighbors(x_query, x_land, g_land, n)
        # neigh (B,n,d), sqdist (B,n), g_gath (B,n)
        r = torch.sqrt(sqdist + self.cfg.eps ** 2)                 # (B,n)
        w = wendland_c2(r, self.cfg.radius)                        # (B,n)

        # (B, n, d) diff between query and its gathered neighbours
        diff = x_query[:, None, :] - neigh                          # (B,n,d)
        wd = w[:, :, None]                                          # (B,n,1)

        # G and ∇G restricted to gathered pairs.
        G, dG = _compute_kernel(diff, sqdist, d, self.cfg.kernel_type,
                                eps=self.cfg.eps, t=self.cfg.t)

        wG = w * G                                                  # (B,n)
        wdG = wd * dG                                               # (B,n,d)

        if self.cfg.normalize_by_cardinality:
            cardinality = (w > 0.0).sum(dim=1).clamp(min=1.0).detach()   # (B,)
            v_hat = (wG * g_gath).sum(dim=1) / cardinality
            gradv_hat = (wdG * g_gath[:, :, None]).sum(dim=1) / cardinality[:, None]
        else:
            denom = float(M)
            v_hat = (wG * g_gath).sum(dim=1) / denom
            gradv_hat = (wdG * g_gath[:, :, None]).sum(dim=1) / denom

        return v_hat, gradv_hat


# -----------------------------
# L2 - k-nearest-neighbour quadrature
# -----------------------------
@dataclass
class KNNConfig:
    eps: float = 1e-2
    k: int = 32               # number of nearest landmarks per query
    # Normalize by k (True) gives the local-mean estimate; False normalizes by M
    # (raw quadrature) -- then convergence to the global estimator requires k->M.
    normalize_by_k: bool = True
    kernel_type: str = "diffusion"  # 'poisson' | 'diffusion'
    t: float = 1.0                  # diffusion scale (ignored if kernel_type='poisson')


class KNNEstimator(nn.Module):
    """L2: k-nearest-neighbour localized Green quadrature."""

    def __init__(self, cfg: KNNConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, x_query, x_land, g_land) -> Tuple[torch.Tensor, torch.Tensor]:
        B, d = x_query.shape
        M = x_land.shape[0]
        k = min(self.cfg.k, M)

        neigh, sqdist, g_gath = _gather_neighbors(x_query, x_land, g_land, k)
        # (B,k,d), (B,k), (B,k)
        diff = x_query[:, None, :] - neigh                          # (B,k,d)

        G, dG = _compute_kernel(diff, sqdist, d, self.cfg.kernel_type,
                                eps=self.cfg.eps, t=self.cfg.t)

        denom = self.cfg.k if self.cfg.normalize_by_k else float(M)
        v_hat = (G * g_gath).sum(dim=1) / denom
        gradv_hat = (dG * g_gath[:, :, None]).sum(dim=1) / denom

        return v_hat, gradv_hat


# -----------------------------
# Helpers
# -----------------------------
def omega_d(d: int) -> float:
    return 2.0 * (math.pi ** (d / 2.0)) / math.gamma(d / 2.0)


def make_localized_estimator(
    scheme: str,
    eps: float = 1e-2,
    radius: float = 1.0,
    k: int = 32,
    normalize: bool = True,
    max_neighbors: Optional[int] = None,
    kernel_type: str = "diffusion",
    t: float = 1.0,
) -> nn.Module:
    """Construct an L1 or L2 localized estimator.

    scheme: 'compact' (L1) | 'knn' (L2) | 'global' (dense MC fallback).
    kernel_type: 'diffusion' | 'poisson'.
    t: diffusion scale, ignored if kernel_type='poisson'.
    """
    if scheme == "compact":
        cfg = CompactSupportConfig(
            eps=eps,
            radius=radius,
            max_neighbors=max_neighbors if max_neighbors is not None else 64,
            normalize_by_cardinality=normalize,
            kernel_type=kernel_type,
            t=t,
        )
        return CompactSupportEstimator(cfg)
    if scheme == "knn":
        return KNNEstimator(KNNConfig(
            eps=eps, k=k, normalize_by_k=normalize,
            kernel_type=kernel_type, t=t,
        ))
    if scheme == "global":
        from .geometry_estimators import PoissonMCConfig, PoissonMCEstimator

        return PoissonMCEstimator(PoissonMCConfig(eps=eps))
    raise ValueError(f"Unknown localized estimator scheme: {scheme}")
