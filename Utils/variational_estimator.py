"""Variational (Ritz) Poisson estimator — ablation alongside diffusion kernel.

Solves, by online gradient descent, the Dirichlet-constrained energy
    min_v  1/2 ||∇v||² + μ/2 ||v||² − ⟨s, v⟩ ,   v|∂Ω = 0
with v = v_θ a small neural field. Interface `forward(x_query, x_land, g_land)
-> (v, gradv)` identical to the kernel estimators, so it plugs into Poisson_reg.

Convention: the inner θ is a stop-gradient variable. The outer loss only
flows through the *query point* x, not through θ itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


# -----------------------------
# Potential field v_θ
# -----------------------------
class PotentialHead(nn.Module):
    """v_theta : R^d -> R, a small residual-MLP mapping space to scalar potential."""

    def __init__(self, d: int, h: int = 128, layers: int = 3):
        super().__init__()
        net = [nn.Linear(d, h), nn.SiLU()]
        for _ in range(layers - 1):
            net += [nn.Linear(h, h), nn.SiLU()]
        net += [nn.Linear(h, 1)]
        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, d) -> (B,) scalar potential."""
        return self.net(x).squeeze(-1)

    def gradv(self, x: torch.Tensor) -> torch.Tensor:
        """Row gradient of v_θ(x): (B, d) -> (B, d)."""
        x = x.requires_grad_(True)
        v = self.forward(x)
        (g,) = torch.autograd.grad(
            v.sum(), x, create_graph=True, retain_graph=True, only_inputs=True
        )
        return g


# -----------------------------
# Ritz energy functional
# -----------------------------
def ritz_energy(
    v: PotentialHead,
    x_colloc: torch.Tensor,
    s_colloc: torch.Tensor,
    x_bd: Optional[torch.Tensor],
    mu: float = 0.0,
    lam_d: float = 1.0,
) -> torch.Tensor:
    """Scalar Ritz energy (1st-order, Dirichlet via soft penalty).

    v:             PotentialHead, the candidate field.
    x_colloc:      (M, d) collocation / landmark points.
    s_colloc:      (M,) source values at collocation.
    x_bd:          (B_bd, d) or None — boundary (OOD) anchor points.
    mu:            screening mass ≥ 0.
    lam_d:         Dirichlet penalty weight.
    """
    grad_v = v.gradv(x_colloc)                # (M, d)
    v_vals = v.forward(x_colloc)               # (M,)
    M = x_colloc.shape[0]

    # 1/2 ||∇v||² + μ/2 ||v||² − ⟨s, v⟩
    energy = (0.5 / M) * (grad_v ** 2).sum(dim=1).sum()
    energy = energy + (mu * 0.5 / M) * (v_vals ** 2).sum()
    energy = energy - (1.0 / M) * (s_colloc * v_vals).sum()

    # Dirichlet soft penalty on boundary / OOD points
    if x_bd is not None and lam_d > 0.0:
        v_bd = v.forward(x_bd)
        energy = energy + lam_d * (v_bd ** 2).mean()

    return energy


# -----------------------------
# Config + forward
# -----------------------------
@dataclass
class VariationalConfig:
    mu: float = 0.0               # screening mass; 0 = pure Poisson
    h: int = 128                  # hidden width
    layers: int = 3               # total layers (including input)
    inner_steps: int = 5          # K GD steps per outer step
    inner_lr: float = 1e-2        # learning rate for inner GD
    lam_d: float = 1.0            # Dirichlet soft-penalty weight
    boundary_as_ood: bool = True  # use x_tilde as Dirichlet anchors


class VariationalEstimator(nn.Module):
    """Variational (Ritz) Poisson estimator — online inner GD, stop-grad on θ."""

    def __init__(self, cfg: VariationalConfig, d: int):
        super().__init__()
        self.cfg = cfg
        self.v = PotentialHead(d, cfg.h, cfg.layers)

    def forward(self, x_query: torch.Tensor, x_land: torch.Tensor, g_land: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the approximately-optimal v_θ* at query points.

        Returns:
          v:      (B,) potential values at x_query.
          gradv:  (B, d) gradient of the potential at x_query.
        """
        cfg = self.cfg
        x_bd = x_query.detach() if cfg.boundary_as_ood else None

        # Inner solve: K gradient-descent steps on the Ritz energy.
        # g_land is stop-gradient w.r.t. the encoder (see PLAN §A.3).
        s = g_land.detach()
        opt = torch.optim.SGD(self.v.parameters(), lr=cfg.inner_lr)
        self.v.train()
        for _ in range(cfg.inner_steps):
            opt.zero_grad()
            loss = ritz_energy(self.v, x_land.detach(), s, x_bd,
                               mu=cfg.mu, lam_d=cfg.lam_d)
            loss.backward()
            opt.step()

        # Evaluate v and gradv at x_query (retains graph to x_query for D_loss).
        self.v.eval()
        v = self.v(x_query)
        gradv = self.v.gradv(x_query)
        return v, gradv
