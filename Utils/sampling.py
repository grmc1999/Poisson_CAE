"""Reconstruction-sample generation from trained checkpoints.

Given a completed run directory (with model_last.pt + config.yml), this
module loads the trained model, applies one or more corruption modes, and
saves a clean / corrupted / reconstruction comparison grid as a PNG.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from Utils.config import ExperimentConfig, apply_overrides, load_config
from Utils.projectors import CorruptionConfig, CorruptionOperator
from Utils.pipeline import build_components


def load_run_config(run_dir: Path) -> ExperimentConfig:
    """Load the saved config.yml from a completed run directory."""
    return load_config(run_dir / "config.yml")


def corruption_override(
    cfg: ExperimentConfig,
    overrides: Dict[str, Any],
) -> ExperimentConfig:
    """Return *cfg* with train-level corruption fields overridden in-place."""
    dotted = {f"train.corruption_{k}": v for k, v in overrides.items()}
    apply_overrides(cfg, dotted)
    return cfg


def sample_batch(
    model: nn.Module,
    Pi: CorruptionOperator,
    loader,
    device: str,
    n: int = 8,
    seed: int = 42,
) -> Dict[str, Any]:
    """Draw *n* samples and return clean / corrupted / reconstructed tensors.

    Returns dict with keys: clean, corrupt, recon (each (n, d) CPU tensors),
    recon_mse, corrupt_mse.
    """
    model.eval()
    torch.manual_seed(seed)
    batch = next(iter(loader))
    x = batch[0][:n].to(device)
    with torch.no_grad():
        x_tilde = Pi(x)[0]
        x_hat = model(x_tilde)
    return {
        "clean": x.detach().cpu(),
        "corrupt": x_tilde.detach().cpu(),
        "recon": x_hat.detach().cpu(),
        "recon_mse": ((x - x_hat) ** 2).mean().item(),
        "corrupt_mse": ((x - x_tilde) ** 2).mean().item(),
    }


def save_reconstruction_grid(
    result: Dict[str, Any],
    input_dim: int,
    out_dir: Path,
) -> None:
    """Save a recon_samples.png + sample_metrics.json to *out_dir*."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    clean = result["clean"]
    corrupt = result["corrupt"]
    recon = result["recon"]
    n = clean.size(0)

    if input_dim == 784:
        fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
        for i in range(n):
            for j, img in enumerate([clean[i], corrupt[i], recon[i]]):
                ax = axes[i][j]
                ax.imshow(img.view(28, 28).cpu().numpy(), cmap="gray")
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "recon_samples.png", dpi=150)
        plt.close(fig)
    else:
        if input_dim <= 2:
            c_np = clean.numpy()
            cr_np = corrupt.numpy()
            re_np = recon.numpy()
        else:
            data = torch.cat([clean, corrupt, recon], dim=0)
            mu = data.mean(0, keepdim=True)
            _, _, V = torch.svd(data - mu)
            proj = V[:, :2]
            c_np = ((clean - mu) @ proj).numpy()
            cr_np = ((corrupt - mu) @ proj).numpy()
            re_np = ((recon - mu) @ proj).numpy()

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.scatter(c_np[:, 0], c_np[:, 1], c="gray", alpha=0.3, s=8, label="clean")
        ax.scatter(cr_np[:, 0], cr_np[:, 1], c="tab:blue", alpha=0.5, s=8, label="corrupt")
        ax.scatter(re_np[:, 0], re_np[:, 1], c="tab:green", alpha=0.5, s=8, label="recon")
        ax.legend()
        ax.set_title("reconstruction (PCA-2D)" if input_dim > 2 else "reconstruction (2D)")
        fig.tight_layout()
        fig.savefig(out_dir / "recon_samples.png", dpi=150)
        plt.close(fig)

    with open(out_dir / "sample_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"recon_mse": result["recon_mse"], "corrupt_mse": result["corrupt_mse"], "n": n},
            fh,
            indent=2,
        )


def generate_samples(
    run_dir: Path,
    modes: Optional[List[str]] = None,
    device: str = "cpu",
    n: int = 8,
    seed: int = 42,
) -> Dict[str, Any]:
    """End-to-end: load model, sample under each corruption mode, save grids.

    Returns dict mapping mode_name -> {recon_mse, corrupt_mse}.
    """
    cfg = load_run_config(run_dir)
    base_corruption_mode = cfg.train.corruption_mode

    if modes is None:
        modes = [base_corruption_mode]

    model, Pi, _est, loader, _test, input_dim, _task = build_components(cfg, device)

    ckpt_path = run_dir / "model_last.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    results: Dict[str, Any] = {}
    for mode in modes:
        override = {}
        if mode != base_corruption_mode:
            override["mode"] = mode
        if override:
            cfg_mod = load_run_config(run_dir)
            corruption_override(cfg_mod, override)
            _model, Pi_mode, _est2, _loader2, _test2, _it, _tk = build_components(cfg_mod, device)
            _model.load_state_dict(model.state_dict())
            _model.to(device).eval()
            result = sample_batch(_model, Pi_mode, loader, device, n=n, seed=seed)
        else:
            result = sample_batch(model, Pi, loader, device, n=n, seed=seed)

        mode_dir = run_dir / f"samples_{mode}"
        save_reconstruction_grid(result, input_dim, mode_dir)
        results[mode] = {"recon_mse": result["recon_mse"], "corrupt_mse": result["corrupt_mse"]}

    return results
