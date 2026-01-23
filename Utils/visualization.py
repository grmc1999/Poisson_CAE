# =========================
# VISUALIZATION UTILITIES
# =========================
import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt

import torch


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


@torch.no_grad()
def _pca_2d(z: torch.Tensor) -> torch.Tensor:
    """
    z: (N, D)
    returns: (N, 2)
    PCA via SVD; no sklearn dependency.
    """
    z = z.detach()
    z = z - z.mean(dim=0, keepdim=True)
    # SVD on covariance equivalent: take top 2 right singular vectors
    # z = U S V^T -> PC directions = V[:, :2]
    _, _, Vt = torch.linalg.svd(z, full_matrices=False)
    pcs = Vt[:2].T  # (D,2)
    return z @ pcs  # (N,2)


def _save_scatter(
    xy: np.ndarray,
    color: np.ndarray,
    outpath: str,
    title: str,
    cmap: str = "viridis",
    s: int = 8,
) -> None:
    plt.figure()
    plt.scatter(xy[:, 0], xy[:, 1], c=color, s=s, cmap=cmap)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _save_scatter_labels(
    xy: np.ndarray,
    labels: np.ndarray,
    outpath: str,
    title: str,
    s: int = 8,
) -> None:
    plt.figure()
    plt.scatter(xy[:, 0], xy[:, 1], c=labels, s=s, cmap="tab10")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _save_hist(values: np.ndarray, outpath: str, title: str, bins: int = 60) -> None:
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _save_image_grid(x: torch.Tensor, outpath: str, title: str, n: int = 16) -> None:
    """
    x: (B,C,H,W) assumed in [0,1] for display.
    """
    x = x[:n].detach().cpu()
    B, C, H, W = x.shape
    cols = int(math.sqrt(n))
    rows = int(math.ceil(n / cols))

    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        img = x[i]
        if C == 1:
            plt.imshow(img[0], cmap="gray", vmin=0.0, vmax=1.0)
        else:
            plt.imshow(img.permute(1, 2, 0).clamp(0, 1))
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def visualize_representations(
    *,
    encoder,
    poisson_estimator,          # must expose v_and_gradv(zq, zl, g_land) -> (v, gradv)
    corruption_operator,        # Πψ(x): torch.Tensor -> torch.Tensor
    dataloader,
    device: str,
    outdir: str,
    z_dim: int,
    hutchinson_fn,              # hutchinson_fn(encoder, x, z_dim, n_samples, create_graph)->g: (B,)
    hutchinson_samples: int = 1,
    n_batches: int = 4,
    landmarks: int = 128,
) -> None:
    """
    Produces:
      - clean/corrupted image grids
      - PCA scatter colored by labels
      - histograms of ||∇f||, v, ||∇v||
      - PCA scatter colored by v and ||∇v||
    """

    _ensure_dir(outdir)

    encoder.eval()

    # Collect a small eval set
    xs, x_tildes, ys = [], [], []
    for b, batch in enumerate(dataloader):
        if b >= n_batches:
            break
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            x, y = batch, None

        x = x.to(device)
        x_tilde = corruption_operator(x)
        xs.append(x)
        x_tildes.append(x_tilde)
        if y is not None:
            ys.append(y.to(device))

    x = torch.cat(xs, dim=0)
    x_tilde = torch.cat(x_tildes, dim=0)
    y = torch.cat(ys, dim=0) if len(ys) > 0 else None

    # Save image grids
    _save_image_grid(x, os.path.join(outdir, "images_clean.png"), "Clean samples", n=16)
    _save_image_grid(x_tilde, os.path.join(outdir, "images_corrupted.png"), "Corrupted samples (Pi(x))", n=16)

    # Latents
    z = encoder(x)           # f(x)
    z_tilde = encoder(x_tilde)

    # PCA for visualization
    xy = _pca_2d(z).cpu().numpy()
    xy_tilde = _pca_2d(z_tilde).cpu().numpy()

    if y is not None:
        _save_scatter_labels(xy, y.detach().cpu().numpy(), os.path.join(outdir, "pca_f_x_labels.png"), "PCA of f(x) colored by label")
        _save_scatter_labels(xy_tilde, y.detach().cpu().numpy(), os.path.join(outdir, "pca_f_xtilde_labels.png"), "PCA of f(Pi(x)) colored by label")

    # ||∇ f(x)|| (contractive quantity)
    # We compute it on corrupted inputs as well, because that’s your experimental focus (OOD/corruption).
    x_land = x_tilde[:min(landmarks, x_tilde.size(0))].clone().requires_grad_(True)
    g_land = hutchinson_fn(encoder, x_land, z_dim, n_samples=hutchinson_samples, create_graph=True)  # (M,)
    g_vals = g_land.detach().cpu().numpy()
    _save_hist(g_vals, os.path.join(outdir, "hist_gradf_norm.png"), r"Histogram of $||\nabla_x f(x)||_F$ on corrupted landmarks")

    # Poisson v and ∇v in latent space (as in your derivation/implementation)
    # Use latent landmarks zl and compute v(z_tilde) and gradv(z_tilde).
    zl = encoder(x_land)  # (M, z_dim)
    v, gradv = poisson_estimator.v_and_gradv(z_tilde, zl, g_land)  # v:(N,), gradv:(N,z_dim)

    v_np = v.detach().cpu().numpy()
    gradv_norm_np = gradv.detach().norm(dim=1).cpu().numpy()

    _save_hist(v_np, os.path.join(outdir, "hist_v.png"), r"Histogram of $v(x)$ evaluated at $z=f(\Pi(x))$")
    _save_hist(gradv_norm_np, os.path.join(outdir, "hist_gradv_norm.png"), r"Histogram of $||\nabla v||$ in latent space")

    # Scatter colored by v and ||∇v||
    _save_scatter(xy_tilde, v_np, os.path.join(outdir, "pca_f_xtilde_v.png"), r"PCA of $f(\Pi(x))$ colored by $v$")
    _save_scatter(xy_tilde, gradv_norm_np, os.path.join(outdir, "pca_f_xtilde_gradv.png"), r"PCA of $f(\Pi(x))$ colored by $||\nabla v||$")

    print(f"[viz] Saved visualization figures to: {outdir}")
