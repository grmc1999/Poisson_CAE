"""Build a potential estimator from the resolved ExperimentConfig.

Phase 2 switchboard: `run.py` used to hard-code a global Poisson MC estimator,
which is numerically infeasible at high input dimension (d >= ~300, see PLAN.md).
This factory turns the `estimator` block of an `ExperimentConfig` into the
concrete `nn.Module` with the common interface

    forward(x_query, x_land, g_land) -> (v_hat, gradv_hat)

so `run.py` / `main.py` never need to know which solver is active.
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from .config import EstimatorConfig


def build_estimator(cfg: EstimatorConfig, d: int) -> nn.Module:
    """Build the estimator selected by `cfg.scheme`.

    scheme:
      - 'global':      dense Poisson Monte Carlo (reference; low-d only).
      - 'compact':     L1 compact-support (Wendland C^2 window) kernel quadrature.
      - 'knn':         L2 exact-kNN kernel quadrature.
      - 'variational': neural-field Ritz (Galerkine-ADE) solver with Dirichlet BC.

    `d` is the input dimension; required by the variational solver to allocate
    its PotentialHead, and used for the default diffusion scale if `t <= 0`
    (fallback t = d / 4, PLAN.md high-d rule).
    """
    from .localized_estimators import make_localized_estimator

    t = cfg.t if cfg.t > 0 else d / 4.0

    if cfg.scheme == "global":
        from .geometry_estimators import PoissonMCConfig, PoissonMCEstimator

        return PoissonMCEstimator(PoissonMCConfig(eps=cfg.eps))

    if cfg.scheme == "compact":
        return make_localized_estimator(
            "compact",
            eps=cfg.eps,
            radius=cfg.radius,
            max_neighbors=cfg.max_neighbors,
            normalize=cfg.normalize,
            kernel_type=cfg.kernel_type,
            t=t,
        )

    if cfg.scheme == "knn":
        return make_localized_estimator(
            "knn",
            eps=cfg.eps,
            k=cfg.k,
            normalize=cfg.normalize,
            kernel_type=cfg.kernel_type,
            t=t,
        )

    if cfg.scheme == "variational":
        from .variational_estimator import VariationalConfig, VariationalEstimator

        return VariationalEstimator(
            VariationalConfig(
                mu=cfg.mu,
                h=cfg.v_hidden,
                layers=cfg.v_layers,
                inner_steps=cfg.inner_steps,
                inner_lr=cfg.inner_lr,
                lam_d=cfg.lam_d,
            ),
            d=d,
        )

    raise ValueError(
        f"Unknown estimator scheme '{cfg.scheme}' "
        f"(expected 'global' | 'compact' | 'knn' | 'variational')"
    )


def default_diffusion_t(d: int) -> float:
    """Heuristic scale for the heat kernel at dimension d: t = d/4.

    Typical inter-point squared distance ||x-y||^2 ~ d for points whose
    coordinates have unit-scale variance, so isotropic diffusion over that
    scale yields exp(-||x-y||^2 / (d)) ~ O(1) contributions instead of
    underflow (0-values) that breaks the gradient signal.
    """
    return d / 4.0