"""Visualization utilities.

The repo started with 2D toy datasets where we can visualize fields over a grid.
For higher-dimensional inputs, a grid is infeasible; instead, we visualize
*sample-based* projections (PCA to 2D) and color-code points by the scalar
quantities of interest.

We support visualizing:
  - f(x): encoder representation
  - ||∇ f(x)||_F: Jacobian Frobenius norm of the encoder w.r.t. inputs
  - v(x): Poisson potential estimated by the Monte-Carlo Green estimator
  - ∇v(x): gradient of the Poisson potential w.r.t. inputs

For d=2 inputs:
  - we plot contours on a grid + quiver fields.

For d>2 inputs:
  - we plot PCA projections of (i) representation z=f(x) and (ii) input x,
    color-coded by the above scalars; and a quiver plot of the input-gradient
    projected onto the input PCA plane.

No external deps beyond matplotlib + torch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .grad_operations import jacobian_fro_norm


# -----------------------------
# Config
# -----------------------------


@dataclass
class VizConfig:
    # 2D grid
    grid_n: int = 160
    padding: float = 0.75
    # Poisson estimator
    landmarks: int = 256
    # Output
    dpi: int = 160
    # PCA
    pca_max_points: int = 4096  # subsample for faster PCA/plotting


# -----------------------------
# Helpers
# -----------------------------


def _torch_pca2(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PCA to 2D using torch.pca_lowrank.

    Returns:
      coords: (N,2)
      mean: (D,)
      V: (D,2) principal directions
    """
    assert X.dim() == 2
    Xc = X - X.mean(dim=0, keepdim=True)
    # q=2 principal components
    U, S, V = torch.pca_lowrank(Xc, q=2, center=False)
    coords = Xc @ V[:, :2]
    return coords, X.mean(dim=0), V[:, :2]


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
    grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    return grid, X, Y


# -----------------------------
# 2D visualization (grid-based)
# -----------------------------


def visualize_fields_2d(
    *,
    model,
    poisson_reg,
    projector,
    x_batch: torch.Tensor,
    out_dir: str,
    step: int,
    device: str | torch.device = "cpu",
    cfg: Optional[VizConfig] = None,
) -> str:
    cfg = cfg or VizConfig()
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_batch = x_batch.detach().to(device)
    grid, X, Y = _make_2d_grid_from_batch(x_batch, cfg.grid_n, cfg.padding, device)
    grid_req = grid.clone().requires_grad_(True)

    # Apply Πψ to get corrupted points (used in Poisson regularizer).
    x_tilde, _ = projector(grid_req)

    # f(x): first 2 components and norm
    with torch.no_grad():
        z = model.Encoder(grid_req)
    if z.shape[1] == 1:
        f1 = z[:, 0]
        f2 = torch.zeros_like(f1)
    else:
        f1, f2 = z[:, 0], z[:, 1]
    fnorm = z.norm(dim=1)

    # ||∇ f(x)||_F on the grid
    g = jacobian_fro_norm(model.Encoder, grid_req, create_graph=False).detach()

    # v(x) and ∇v(x) using the repo's estimator
    v, gradv = poisson_reg.Estimate_field_grads(grid_req, x_tilde.detach(), landmarks=cfg.landmarks)
    v = v.detach()
    gradv = gradv.detach()

    def R(u: torch.Tensor) -> torch.Tensor:
        return u.reshape(cfg.grid_n, cfg.grid_n)

    v_img = R(v).cpu()
    g_img = R(g).cpu()
    fn_img = R(fnorm).cpu()

    # Quiver downsample
    q = max(1, cfg.grid_n // 25)
    Xc = X[::q, ::q].cpu()
    Yc = Y[::q, ::q].cpu()
    gvx = gradv[:, 0].reshape(cfg.grid_n, cfg.grid_n)[::q, ::q].cpu()
    gvy = gradv[:, 1].reshape(cfg.grid_n, cfg.grid_n)[::q, ::q].cpu()
    f1q = f1.reshape(cfg.grid_n, cfg.grid_n)[::q, ::q].cpu()
    f2q = f2.reshape(cfg.grid_n, cfg.grid_n)[::q, ::q].cpu()

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
    gradv_norm = torch.sqrt((gradv ** 2).sum(dim=1) + 1e-12)
    im = ax.contourf(X.cpu(), Y.cpu(), R(gradv_norm).cpu(), levels=40)
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


# -----------------------------
# d>2 visualization (sample-based)
# -----------------------------


def visualize_fields_nd(
    *,
    model,
    poisson_reg,
    projector,
    x_batch: torch.Tensor,
    out_dir: str,
    step: int,
    device: str | torch.device = "cpu",
    cfg: Optional[VizConfig] = None,
) -> str:
    """Visualize fields for d>2 inputs using PCA projections.

    Panels (2x2):
      (0,0) PCA(z=f(x)) colored by ||f||
      (0,1) PCA(z) colored by ||∇f||_F
      (1,0) PCA(x) colored by v(x)
      (1,1) PCA(x) quiver of projected ∇v(x) and color by ||∇v||
    """
    cfg = cfg or VizConfig()
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = x_batch.detach().to(device)
    if x.dim() != 2:
        # If user passes sequences/images flattened elsewhere, keep this strict.
        x = x.view(x.size(0), -1)

    # Subsample for visualization/PCA
    N = x.size(0)
    if N > cfg.pca_max_points:
        idx = torch.randperm(N, device=device)[: cfg.pca_max_points]
        x = x[idx]

    x_req = x.clone().requires_grad_(True)
    x_tilde, _ = projector(x_req)

    # Representation
    with torch.no_grad():
        z = model.Encoder(x_req)
    fnorm = z.norm(dim=1)

    # ||∇f||_F in input space
    g = jacobian_fro_norm(model.Encoder, x_req, create_graph=False).detach()

    # v and ∇v (input space)
    v, gradv = poisson_reg.Estimate_field_grads(x_req, x_tilde.detach(), landmarks=cfg.landmarks)
    v = v.detach()
    gradv = gradv.detach()
    gradv_norm = torch.sqrt((gradv ** 2).sum(dim=1) + 1e-12)

    # PCA projections
    z2, _, _ = _torch_pca2(z.detach())
    x2, _, Vx = _torch_pca2(x.detach())

    # Project ∇v onto PCA(x) plane: (N,d) @ (d,2)
    gradv_2 = gradv @ Vx  # (N,2)

    z2 = z2.cpu()
    x2 = x2.cpu()
    fnorm_c = fnorm.cpu()
    g_c = g.cpu()
    v_c = v.cpu()
    gradv_norm_c = gradv_norm.cpu()
    gradv_2 = gradv_2.cpu()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=cfg.dpi)

    ax = axes[0, 0]
    sc = ax.scatter(z2[:, 0], z2[:, 1], c=fnorm_c, s=6, alpha=0.7)
    ax.set_title("PCA(z=f(x)) colored by ||f||")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax = axes[0, 1]
    sc = ax.scatter(z2[:, 0], z2[:, 1], c=g_c, s=6, alpha=0.7)
    ax.set_title(r"PCA(z) colored by $||\nabla f||_F$")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax = axes[1, 0]
    sc = ax.scatter(x2[:, 0], x2[:, 1], c=v_c, s=6, alpha=0.7)
    ax.set_title("PCA(x) colored by v(x)")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax = axes[1, 1]
    # downsample quiver for readability
    npts = x2.size(0)
    q = max(1, npts // 250)
    idx = torch.arange(0, npts, q)
    sc = ax.scatter(x2[:, 0], x2[:, 1], c=gradv_norm_c, s=6, alpha=0.55)
    ax.quiver(
        x2[idx, 0],
        x2[idx, 1],
        gradv_2[idx, 0],
        gradv_2[idx, 1],
        angles="xy",
        scale_units="xy",
        scale=None,
        width=0.002,
        alpha=0.8,
    )
    ax.set_title(r"PCA(x): projected $\nabla v$ (quiver), colored by $||\nabla v||$")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    fig.suptitle(f"Poisson-CAE diagnostics (d={x_batch.shape[1]}) @ step {step}", y=1.02)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"fields_step_{step:06d}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# -----------------------------
# Public entrypoint
# -----------------------------


def visualize_fields(
    *,
    model,
    poisson_reg,
    projector,
    x_batch: torch.Tensor,
    out_dir: str,
    step: int,
    device: str | torch.device = "cpu",
    cfg: Optional[VizConfig] = None,
) -> str:
    """Dispatch to 2D or ND visualization."""
    if x_batch.dim() == 2 and x_batch.shape[1] == 2:
        return visualize_fields_2d(
            model=model,
            poisson_reg=poisson_reg,
            projector=projector,
            x_batch=x_batch,
            out_dir=out_dir,
            step=step,
            device=device,
            cfg=cfg,
        )
    return visualize_fields_nd(
        model=model,
        poisson_reg=poisson_reg,
        projector=projector,
        x_batch=x_batch,
        out_dir=out_dir,
        step=step,
        device=device,
        cfg=cfg,
    )
