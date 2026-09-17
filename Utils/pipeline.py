"""Shared model/data/estimator construction for training and sampling.

Separated from run.py so both the training loop and the sampler can build
the same components from a config without duplication.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from Utils.config import ExperimentConfig
from Utils.datasets import LoaderCfg, get_experiment_loaders
from Utils.estimator_factory import build_estimator
from Utils.projectors import CorruptionConfig, CorruptionOperator

from models import (
    AE_model,
    Classifier_model,
    Encoder,
    Decoder,
    GRUEncoder,
    Regressor_model,
)


def build_corruption_operator(cfg: ExperimentConfig) -> CorruptionOperator:
    """Build the corruption operator Pi from a config.

    rotation/zoom operate on the spatial layout; MNIST flat is seen as 28x28.
    """
    image_side = 28 if cfg.data.experiment == "mnist_flat" else 0
    return CorruptionOperator(
        CorruptionConfig(
            mode=cfg.train.corruption_mode,
            T=cfg.train.corruption_T,
            beta_start=cfg.train.corruption_beta_start,
            beta_end=cfg.train.corruption_beta_end,
            sigma=cfg.train.corruption_sigma,
            mask_frac=cfg.train.corruption_mask_frac,
            drop_p=cfg.train.corruption_drop_p,
            rotation_max_deg=cfg.train.corruption_rotation_max_deg,
            zoom_std=cfg.train.corruption_zoom_std,
            image_side=image_side,
        )
    )


def build_components(
    cfg: ExperimentConfig,
    device: str,
) -> Tuple[
    torch.nn.Module,
    CorruptionOperator,
    torch.nn.Module,
    DataLoader,
    Optional[DataLoader],
    int,
    str,
]:
    """Build model, corruption operator, estimator, and data loaders.

    Returns (model, Pi, estimator, loader, test_loader, input_dim, task).
    """
    torch.manual_seed(cfg.data.seed)

    # ---- data -----------------------------------------------------------
    if cfg.data.experiment == "mog":
        N = 5000
        centers = torch.tensor(
            [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.25]], dtype=torch.float32
        )
        comp = torch.randint(0, centers.size(0), (N,))
        x = centers[comp] + 0.15 * torch.randn(N, 2)
        loader = DataLoader(
            TensorDataset(x),
            batch_size=cfg.data.batch_size,
            shuffle=True,
            drop_last=True,
        )
        test_loader = None
        input_dim = 2
        task = "reconstruction"
    else:
        loader, test_loader, input_dim, task = get_experiment_loaders(
            cfg.data.experiment,
            LoaderCfg(
                batch_size=cfg.data.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
            ),
            seed=cfg.data.seed,
        )

    # ---- model ----------------------------------------------------------
    if task == "reconstruction":
        model = AE_model(
            Encoder(d=input_dim, h=cfg.model.hidden, z=cfg.model.z_dim),
            Decoder(z=cfg.model.z_dim, h=cfg.model.hidden, d=input_dim),
        )
    elif task == "classification":
        model = Classifier_model(
            Encoder(d=input_dim, h=cfg.model.hidden, z=cfg.model.z_dim),
            n_classes=2,
        )
    elif task == "regression":
        if cfg.data.encoder_type == "gru":
            enc = GRUEncoder(
                T=input_dim, din=1, hidden=cfg.model.hidden, z_dim=cfg.model.z_dim
            )
        else:
            enc = Encoder(d=input_dim, h=cfg.model.hidden, z=cfg.model.z_dim)
        model = Regressor_model(enc, out_dim=3)
    else:
        raise ValueError(f"Unknown task: {task}")

    # ---- corruption operator --------------------------------------------
    Pi = build_corruption_operator(cfg)

    # ---- estimator ------------------------------------------------------
    estimator = build_estimator(cfg.estimator, d=input_dim)

    return model, Pi, estimator, loader, test_loader, input_dim, task
