"""Visualization utilities.

This project studies geometric/Poisson regularization terms. For quick
sanity-checks we provide simple plotting functions for 2D toy datasets.

We visualize:
  - f(x): encoder representation (first 2 components + norm)
  - ||∇ f(x)||_F: Jacobian Frobenius norm of the encoder
  - v(x): Poisson potential estimated by the Monte Carlo Green's estimator
  - ∇v(x): gradient field of the Poisson potential

These functions are intentionally lightweight and depend only on matplotlib.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .grad_operations import jacobian_fro_norm
from sklearn.decomposition import PCA


@dataclass
class Viz2DConfig:
    grid_n: int = 160
    padding: float = 0.75
    landmarks: int = 256
    dpi: int = 160


def _make_2d_grid_from_batch(
    x_batch: torch.Tensor,
    grid_n: int,
    padding: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a 2D grid covering the batch extent."""
    assert x_batch.shape[1] == 2, "2D visualization expects d=2 inputs"

    x_min = x_batch.min(dim=0).values - padding
    x_max = x_batch.max(dim=0).values + padding

    xs = torch.linspace(x_min[0].item(), x_max[0].item(), grid_n, device=device)
    ys = torch.linspace(x_min[1].item(), x_max[1].item(), grid_n, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)  # (grid_n^2, 2)
    return grid, X, Y


def visualize_fields_2d(
    *,
    model,
    poisson_reg,
    projector,
    x_batch: torch.Tensor,
    out_dir: str,
    step: int,
    device: str | torch.device = "cpu",
    cfg: Optional[Viz2DConfig] = None,
) -> str:
    """Save a 2D visualization of f, ||∇f||, v and ∇v.

    Parameters
    ----------
    model:
        AE_model (must expose model.Encoder)
    poisson_reg:
        Poisson_reg instance (must expose Estimate_field_grads)
    projector:
        CorruptionOperator (Πψ) returning (x_tilde, t)
    x_batch:
        (B,2) batch of data
    out_dir:
        directory to save figures
    step:
        training step (used in filename)

    Returns
    -------
    Path to the saved PNG.
    """
    cfg = cfg or Viz2DConfig()
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device)

    # Lazy import to keep core dependencies minimal.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_batch = x_batch.detach().to(device)
    grid, X, Y = _make_2d_grid_from_batch(x_batch, cfg.grid_n, cfg.padding, device)
    grid_req = grid.clone().requires_grad_(True)

    # Apply Πψ to get corrupted points (used in Poisson regularizer).
    x_tilde, _ = projector(grid_req)

    # f(x): take first 2 dims (if z<2, fall back to repeating)
    with torch.no_grad():
        z = model.Encoder(grid_req)
    if z.shape[1] == 1:
        f1 = z[:, 0]
        f2 = torch.zeros_like(f1)
    else:
        f1, f2 = z[:, 0], z[:, 1]
        T_p = PCA(n_components=2).fit(z)
        zp=T_p.transform(z)
        f1, f2 = zp[:, 0], zp[:, 1]
    fnorm = z.norm(dim=1)

    # ||∇ f(x)||_F on the grid
    g = jacobian_fro_norm(model.Encoder, grid_req, create_graph=False).detach()

    # v(x) and ∇v(x) using the repo's estimator
    v, gradv = poisson_reg.Estimate_field_grads(grid_req, x_tilde.detach(), landmarks=cfg.landmarks)
    v = v.detach()
    gradv = gradv.detach()

    # Reshape scalars to (grid_n, grid_n)
    def R(u: torch.Tensor) -> torch.Tensor:
        return u.reshape(cfg.grid_n, cfg.grid_n)

    v_img = R(v).cpu()
    g_img = R(g).cpu()
    f1_img, f2_img = R(f1), R(f2)
    fn_img = R(fnorm)

    # Quiver downsample factor (avoid too many arrows)
    q = max(1, cfg.grid_n // 25)
    Xc = X[::q, ::q].cpu()
    Yc = Y[::q, ::q].cpu()
    gvx = gradv[:, 0].reshape(cfg.grid_n, cfg.grid_n)[::q, ::q].cpu()
    gvy = gradv[:, 1].reshape(cfg.grid_n, cfg.grid_n)[::q, ::q].cpu()
    f1q = f1.reshape(cfg.grid_n, cfg.grid_n)[::q, ::q]#.cpu()
    f2q = f2.reshape(cfg.grid_n, cfg.grid_n)[::q, ::q]#.cpu()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=cfg.dpi)
    ax = axes[0, 0]
    im = ax.contourf(X.cpu(), Y.cpu(), fn_img, levels=40)
    ax.quiver(Xc, Yc, f1q, f2q, angles="xy", scale_units="xy", scale=None, width=0.002)
    ax.scatter(x_batch[:, 0].cpu(), x_batch[:, 1].cpu(), s=4, alpha=0.35)
    ax.set_title("f(x): ||f|| (contour) + first-2 comps (quiver)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    im = ax.contourf(X.cpu(), Y.cpu(), g_img, levels=40)
    ax.scatter(x_batch[:, 0].cpu(), x_batch[:, 1].cpu(), s=4, alpha=0.35)
    ax.set_title(r"$||\nabla f(x)||_F$ (contour)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    im = ax.contourf(X.cpu(), Y.cpu(), v_img, levels=40)
    ax.scatter(x_batch[:, 0].cpu(), x_batch[:, 1].cpu(), s=4, alpha=0.35)
    ax.set_title("v(x) (Poisson potential; contour)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    speed = torch.sqrt(gvx**2 + gvy**2)
    im = ax.contourf(X.cpu(), Y.cpu(), R(torch.sqrt((gradv**2).sum(dim=1))), levels=40)
    ax.quiver(Xc, Yc, gvx, gvy, angles="xy", scale_units="xy", scale=None, width=0.002)
    ax.scatter(x_batch[:, 0].cpu(), x_batch[:, 1].cpu(), s=4, alpha=0.35)
    ax.set_title(r"$\nabla v(x)$ (quiver) and $||\nabla v||$ (contour)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes.reshape(-1):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    fig.suptitle(f"Poisson-CAE fields @ step {step}", y=1.02)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"fields_step_{step:06d}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
