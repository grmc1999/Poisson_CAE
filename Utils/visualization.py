"""Visualization utilities.

This repo started with 2D toy experiments. For MNIST we visualize:
  - f(x): latent codes and norms
  - ||∇f(x)||: contractive source term histogram
  - v(x): Poisson potential histogram
  - ∇v: in pixel space (flattened) we show ||∇v|| as an image heatmap

All functions save PNGs to disk (no interactive windows), keeping main scripts clean.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class VizConfig:
    out_dir: str = "outputs"
    max_items: int = 8
    dpi: int = 120


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def visualize_mnist_flat(
    x_clean: torch.Tensor,         # (B, 784)
    x_corrupt: torch.Tensor,       # (B, 784)
    x_hat: torch.Tensor,           # (B, 784)
    *,
    f: torch.Tensor,               # (B, z)
    g: torch.Tensor,               # (B,)
    v: torch.Tensor,               # (B,)
    gradv: Optional[torch.Tensor], # (B, 784)
    step: int,
    cfg: VizConfig = VizConfig(),
    title: str = "mnist_flat",
) -> str:
    """Save a multi-panel figure for flattened MNIST."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(cfg.out_dir)
    B = x_clean.size(0)
    K = min(cfg.max_items, B)

    x0 = x_clean[:K].view(K, 28, 28).cpu()
    xc = x_corrupt[:K].view(K, 28, 28).cpu()
    xr = x_hat[:K].view(K, 28, 28).cpu()

    fig = plt.figure(figsize=(12, 8), dpi=cfg.dpi)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.0])

    # Row 1: images
    ax = fig.add_subplot(gs[0, 0]); ax.set_title("clean")
    ax.imshow(torch.cat(list(x0), dim=1), cmap="gray"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 1]); ax.set_title("corrupted")
    ax.imshow(torch.cat(list(xc), dim=1), cmap="gray"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 2]); ax.set_title("recon")
    ax.imshow(torch.cat(list(xr), dim=1), cmap="gray"); ax.axis("off")

    # Row 1 col 4: ||∇v|| heatmap for first sample
    ax = fig.add_subplot(gs[0, 3]); ax.set_title("||∇v(x)|| (sample 0)")
    if gradv is None:
        ax.text(0.1, 0.5, "gradv=None", fontsize=10)
        ax.axis("off")
    else:
        gv = gradv[0].view(28, 28).abs().cpu()
        ax.imshow(gv, cmap="magma"); ax.axis("off")

    # Row 2: scalar histograms
    ax = fig.add_subplot(gs[1, 0]); ax.set_title("||f(x)||")
    ax.hist(f.norm(dim=1).detach().cpu().numpy(), bins=30)
    ax = fig.add_subplot(gs[1, 1]); ax.set_title("||∇f(x)||_F")
    ax.hist(g.detach().cpu().numpy(), bins=30)
    ax = fig.add_subplot(gs[1, 2]); ax.set_title("v(x)")
    ax.hist(v.detach().cpu().numpy(), bins=30)

    # Row 2 col 4: scatter of first 2 latent dims (if available)
    ax = fig.add_subplot(gs[1, 3]); ax.set_title("f(x) (first 2 dims)")
    if f.size(1) >= 2:
        z = f.detach().cpu()
        ax.scatter(z[:, 0], z[:, 1], s=5, alpha=0.6)
    else:
        ax.text(0.1, 0.5, "latent dim < 2", fontsize=10)

    # Row 3: per-sample values table-ish
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    vals = []
    for i in range(min(10, B)):
        vals.append(f"i={i}: ||f||={f[i].norm().item():.3f},  g={g[i].item():.3f},  v={v[i].item():.3f}")
    ax.text(0.01, 0.95, "\n".join(vals), va="top", family="monospace")

    fig.suptitle(f"{title} | step={step}")
    out_path = os.path.join(cfg.out_dir, f"{title}_step_{step:06d}.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


@torch.no_grad()
def visualize_mnist_conv(
    x_clean: torch.Tensor,         # (B,1,28,28)
    x_corrupt: torch.Tensor,       # (B,1,28,28)
    x_hat: torch.Tensor,           # (B,1,28,28)
    *,
    z: torch.Tensor,               # (B, zdim) clean
    z_tilde: torch.Tensor,         # (B, zdim) corrupted
    g: torch.Tensor,               # (B,) source term
    v: torch.Tensor,               # (B,) potential
    gradv: torch.Tensor,           # (B, zdim)
    step: int,
    cfg: VizConfig = VizConfig(),
    title: str = "mnist_conv",
) -> str:
    """Save a multi-panel figure for conv MNIST (latent-space Poisson)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(cfg.out_dir)
    B = x_clean.size(0)
    K = min(cfg.max_items, B)

    x0 = x_clean[:K].squeeze(1).cpu()
    xc = x_corrupt[:K].squeeze(1).cpu()
    xr = x_hat[:K].squeeze(1).cpu()

    fig = plt.figure(figsize=(12, 8), dpi=cfg.dpi)
    gs = fig.add_gridspec(3, 4)

    ax = fig.add_subplot(gs[0, 0]); ax.set_title("clean")
    ax.imshow(torch.cat(list(x0), dim=1), cmap="gray"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 1]); ax.set_title("corrupted")
    ax.imshow(torch.cat(list(xc), dim=1), cmap="gray"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 2]); ax.set_title("recon")
    ax.imshow(torch.cat(list(xr), dim=1), cmap="gray"); ax.axis("off")

    # Show ||gradv|| per-sample (latent), not pixel
    ax = fig.add_subplot(gs[0, 3]); ax.set_title("||∇v(z)||")
    ax.hist(gradv.norm(dim=1).detach().cpu().numpy(), bins=30)

    # Histograms
    ax = fig.add_subplot(gs[1, 0]); ax.set_title("||z||")
    ax.hist(z.norm(dim=1).detach().cpu().numpy(), bins=30)
    ax = fig.add_subplot(gs[1, 1]); ax.set_title("||z_tilde-z||")
    ax.hist((z_tilde - z).norm(dim=1).detach().cpu().numpy(), bins=30)
    ax = fig.add_subplot(gs[1, 2]); ax.set_title("||∇f|| (Hutchinson)")
    ax.hist(g.detach().cpu().numpy(), bins=30)
    ax = fig.add_subplot(gs[1, 3]); ax.set_title("v(z_tilde)")
    ax.hist(v.detach().cpu().numpy(), bins=30)

    # Scatter first 2 dims
    ax = fig.add_subplot(gs[2, :2]); ax.set_title("z (first 2 dims)")
    if z.size(1) >= 2:
        zz = z.detach().cpu()
        ax.scatter(zz[:, 0], zz[:, 1], s=5, alpha=0.6)
    else:
        ax.text(0.1, 0.5, "latent dim < 2", fontsize=10)

    ax = fig.add_subplot(gs[2, 2:]); ax.axis("off")
    vals = []
    for i in range(min(10, B)):
        vals.append(
            f"i={i}: ||z||={z[i].norm().item():.3f},  g={g[i].item():.3f},  v={v[i].item():.3f},  ||∇v||={gradv[i].norm().item():.3f}"
        )
    ax.text(0.01, 0.95, "\n".join(vals), va="top", family="monospace")

    fig.suptitle(f"{title} | step={step}")
    out_path = os.path.join(cfg.out_dir, f"{title}_step_{step:06d}.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
