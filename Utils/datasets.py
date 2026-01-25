"""Dataset generators and loaders for low-dimensional experiments.

The project targets *low-dimensional* (<=50) unstructured data to study
contractive regularization and its Poisson reformulation.

Implemented experiments (as requested):
  1) 2D spirals (classification)
  2) banana / two-moons (classification)
  3) concentric rings (classification)
  4) breast cancer Wisconsin (classification, 30D)
  5) time-series vectors (regression or reconstruction; <=50 dims)

All returned tensors are torch.float32, with labels as torch.long.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


try:
    from sklearn.datasets import make_moons, make_circles, load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder
except Exception:  # pragma: no cover
    make_moons = make_circles = load_breast_cancer = None
    train_test_split = None
    StandardScaler = None


@dataclass
class LoaderCfg:
    batch_size: int = 256
    shuffle: bool = True
    drop_last: bool = True
    num_workers: int = 0


def _to_loader(x: torch.Tensor, y: Optional[torch.Tensor], cfg: LoaderCfg) -> DataLoader:
    if y is None:
        ds = TensorDataset(x)
    else:
        ds = TensorDataset(x, y)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
    )


# -----------------------------
# 2D synthetic datasets
# -----------------------------
def make_two_spirals(
    n: int = 5000,
    noise: float = 0.2,
    revolutions: float = 3.0,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Two intertwined spirals in R^2.

    Returns:
      x: (n,2) float32
      y: (n,) long in {0,1}
    """
    rng = np.random.default_rng(seed)
    n2 = n // 2
    theta = np.linspace(0.0, 2.0 * np.pi * revolutions, n2)
    r = theta
    x1 = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    x2 = np.stack([r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)], axis=1)
    x = np.concatenate([x1, x2], axis=0)
    x = x / (x.std(axis=0, keepdims=True) + 1e-8)
    x = x + noise * rng.standard_normal(size=x.shape)
    y = np.concatenate([np.zeros(n2, dtype=np.int64), np.ones(n - n2, dtype=np.int64)], axis=0)
    # shuffle
    perm = rng.permutation(n)
    x, y = x[perm], y[perm]
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y)


def make_banana_moons(
    n: int = 5000,
    noise: float = 0.15,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Two-moons (banana) dataset."""
    if make_moons is None:
        raise RuntimeError("scikit-learn is required for make_moons")
    x, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    x = x.astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))


def make_rings(
    n: int = 5000,
    noise: float = 0.08,
    factor: float = 0.5,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concentric rings dataset (two circles)."""
    if make_circles is None:
        raise RuntimeError("scikit-learn is required for make_circles")
    x, y = make_circles(n_samples=n, noise=noise, factor=factor, random_state=seed)
    x = x.astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))


# -----------------------------
# Breast cancer Wisconsin (30D)
# -----------------------------
def make_breast_cancer(
    test_size: float = 0.2,
    seed: int = 0,
    standardize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Loads and splits Breast Cancer Wisconsin dataset.

    Returns:
      x_train, y_train, x_test, y_test
    """
    if load_breast_cancer is None or train_test_split is None:
        raise RuntimeError("scikit-learn is required for breast cancer dataset")
    data = load_breast_cancer()
    x = data.data.astype(np.float32)
    y = data.target.astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)

    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)
    if standardize:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).astype(np.float32)
        x_test = scaler.transform(x_test).astype(np.float32)
    return (
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(x_test),
        torch.from_numpy(y_test),
    )


# -----------------------------
# Time-series vectors (<=50 dims)
# -----------------------------
def make_sinusoid_params_regression(
    n: int = 8000,
    T: int = 50,
    seed: int = 0,
    noise: float = 0.05,
    freq_range: Tuple[float, float] = (1.0, 6.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate 1D sinusoid time-series (flattened to R^T) with targets (A, f, phi).

    x_i[t] = A * sin(2π f t/T + phi) + noise
    y_i = [A, f, phi]
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float32)[None, :]
    A = rng.uniform(0.5, 1.5, size=(n, 1)).astype(np.float32)
    f = rng.uniform(freq_range[0], freq_range[1], size=(n, 1)).astype(np.float32)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(n, 1)).astype(np.float32)
    x = A * np.sin(2.0 * np.pi * f * t / float(T) + phi)
    x = x + noise * rng.standard_normal(size=x.shape).astype(np.float32)
    y = np.concatenate([A, f, phi], axis=1).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


DatasetName = Literal[
    "spirals",
    "banana",
    "rings",
    "breast_cancer",
    "sinusoid_reg",
]


def get_experiment_loaders(
    name: DatasetName,
    cfg: LoaderCfg,
    seed: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader], int, str]:
    """Return (train_loader, test_loader_or_None, input_dim, task_type).

    task_type is one of:
      - "classification"
      - "regression"
      - "reconstruction"

    For 2D patterns, we return only train_loader by default (easy to visualize).
    For breast_cancer and sinusoid_reg, we provide train+test.
    """
    name = str(name).lower()
    if name == "spirals":
        x, y = make_two_spirals(n=6000, noise=0.15, seed=seed)
        return _to_loader(x, y, cfg), None, x.shape[1], "classification"
    if name == "banana":
        x, y = make_banana_moons(n=6000, noise=0.18, seed=seed)
        return _to_loader(x, y, cfg), None, x.shape[1], "classification"
    if name == "rings":
        x, y = make_rings(n=6000, noise=0.06, seed=seed)
        return _to_loader(x, y, cfg), None, x.shape[1], "classification"
    if name == "breast_cancer":
        xtr, ytr, xte, yte = make_breast_cancer(seed=seed)
        tr = _to_loader(xtr, ytr, cfg)
        te = _to_loader(xte, yte, LoaderCfg(batch_size=cfg.batch_size, shuffle=False, drop_last=False))
        return tr, te, xtr.shape[1], "classification"
    if name == "sinusoid_reg":
        x, y = make_sinusoid_params_regression(n=10000, T=50, seed=seed)
        # split
        n = x.shape[0]
        idx = torch.randperm(n)
        x = x[idx]; y = y[idx]
        split = int(0.8 * n)
        xtr, ytr = x[:split], y[:split]
        xte, yte = x[split:], y[split:]
        tr = _to_loader(xtr, ytr, cfg)
        te = _to_loader(xte, yte, LoaderCfg(batch_size=cfg.batch_size, shuffle=False, drop_last=False))
        return tr, te, x.shape[1], "regression"

    raise ValueError(f"Unknown experiment name: {name}")
