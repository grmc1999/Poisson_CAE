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

    def reset_parameters(self) -> None:
        """Re-initialize like a freshly-constructed head (used after bailouts)."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

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
    mu: float = 1e-2              # screening mass; stiffens the inner solve
    h: int = 128                  # hidden width
    layers: int = 3               # total layers (including input)
    inner_steps: int = 5          # K GD steps per outer step
    inner_lr: float = 1e-2        # learning rate for inner GD
    lam_d: float = 1.0            # Dirichlet soft-penalty weight
    boundary_as_ood: bool = True  # use x_tilde as Dirichlet anchors
    inner_max_grad_norm: float = 0.5  # inner-step grad-norm clip (0 = off)
    bilevel: bool = True          # differentiate through the inner solve (BOP)


CMG_ITERS = 25  # conjugate-gradient iterations for the BOP Hessian solve


def _flat_blocks(vec: torch.Tensor, shapes) -> list[torch.Tensor]:
    """Split a flat vector back into per-parameter blocks matching `shapes`."""
    blocks, offset = [], 0
    for shp in shapes:
        n = 1
        for s in shp:
            n *= s
        blocks.append(vec[offset : offset + n].view(shp))
        offset += n
    return blocks


def cg_solve(hvp, b: torch.Tensor, iters: int = CMG_ITERS) -> torch.Tensor:
    """Conjugate-gradient solve H·u = b using Hessian-vector products."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    r_dot = torch.dot(r, r)
    if r_dot.item() == 0:
        return x
    for _ in range(iters):
        ap = hvp(p)
        alpha = r_dot / (torch.dot(p, ap) + 1e-12)
        x = x + alpha * p
        r = r - alpha * ap
        r_new = torch.dot(r, r)
        if r_new < 1e-24:
            break
        beta = r_new / (r_dot + 1e-12)
        p = r + beta * p
        r_dot = r_new
    return x


class BOPPoissonSolve(torch.autograd.Function):
    """Bilevel (BOP) variational Poisson solve.

    forward :  run the K-step inner GD on the Ritz energy (θ = PotentialHead
               params, source `s` NOT stopped), then evaluate the field at the
               queries with θ frozen.
    backward:  implicit-function differentiation of the inner minimizer
               θ*(s):  dL/ds = -∇_s ⟨g, u⟩  with  H·u = b,  H = ∇²_θ E,
               g = ∇_θ E at the optimum, b = gradient of the outer loss
               through the field into θ. Also returns df/dx_query.
    """

    @staticmethod
    def forward(ctx, x_query, x_land, s, head: "PotentialHead", cfg: VariationalConfig):
        # Custom Function.forward runs with grad DISABLED by the engine; the IFT
        # machinery needs real (double) backward graphs, so re-enable grad here.
        # The Function boundary detaches internals, so this cannot leak grads into θ.
        with torch.enable_grad():
            v, gradv = BOPPoissonSolve._forward_graph(
                ctx, x_query, x_land, s, head, cfg
            )
        return v, gradv

    @staticmethod
    def _forward_graph(ctx, x_query, x_land, s, head, cfg):
        theta = list(head.parameters())
        x_bd = x_query.detach() if cfg.boundary_as_ood else None

        # 1) Inner solve: K SGD steps, plain (no unrolled graph needed; IFT in backward).
        #    Use autograd.grad(., theta) so the inner step NEVER backprops into the
        #    encoder through s=g_land (which would contaminate the outer model grads).
        for _ in range(cfg.inner_steps):
            energy = ritz_energy(head, x_land.detach(), s, x_bd, mu=cfg.mu, lam_d=cfg.lam_d)
            grads = torch.autograd.grad(energy, theta, retain_graph=False)
            if cfg.inner_max_grad_norm > 0 and len(grads) > 0:
                gn = sum((gi * gi).sum() for gi in grads).sqrt()
                scale = cfg.inner_max_grad_norm / (gn + 1e-8)
                if float(scale.detach()) < 1.0:
                    grads = [gi * scale for gi in grads]
            with torch.no_grad():
                for p, g in zip(theta, grads):
                    p.sub_(cfg.inner_lr * g)

        # 2) Stationarity gradient g = ∇_θ E at the converged θ (keeps s-graph).
        energy_final = ritz_energy(head, x_land.detach(), s, x_bd, mu=cfg.mu, lam_d=cfg.lam_d)
        g_final = torch.autograd.grad(
            energy_final, theta, create_graph=True, retain_graph=True, allow_unused=True
        )

        # 3) Evaluate field at queries. The Function detaches internals, so this
        #    only carries dependence on x_query (data) into the outputs; the model
        #    dependence is delivered in backward() via the IFT gradient through s.
        v = head(x_query)
        gradv = head.gradv(x_query)
        for p in theta:
            p.grad = None

        ctx.head = head
        ctx.cfg = cfg
        ctx.theta = theta
        ctx.g_final = g_final
        ctx.save_for_backward(x_query, s)
        return v, gradv

    @staticmethod
    def backward(ctx, grad_v, grad_gradv):
        with torch.enable_grad():
            return BOPPoissonSolve._backward_graph(ctx, grad_v, grad_gradv)

    @staticmethod
    def _backward_graph(ctx, grad_v, grad_gradv):
        head = ctx.head
        cfg = ctx.cfg
        theta = ctx.theta
        (x_query, s) = ctx.saved_tensors
        xq = x_query.detach()
        xq.requires_grad_(True)

        for p in theta:
            p.requires_grad_(True)

        # Re-evaluate the final field at final θ to get its θ-gradients.
        v = head(xq)
        gradv = head.gradv(xq)
        b_blocks = torch.autograd.grad(
            v, theta, grad_outputs=grad_v, retain_graph=True, allow_unused=True
        )
        b2_blocks = torch.autograd.grad(
            gradv, theta, grad_outputs=grad_gradv, retain_graph=True, allow_unused=True
        )
        # Combine the v- and gradv- vjps into one per-parameter gradient b.
        combined = []
        for g1, g2, p in zip(b_blocks, b2_blocks, theta):
            if g1 is None and g2 is None:
                combined.append(torch.zeros_like(p).reshape(-1))
            elif g1 is None:
                combined.append(g2.reshape(-1))
            elif g2 is None:
                combined.append(g1.reshape(-1))
            else:
                combined.append((g1 + g2).reshape(-1))
        b_flat = torch.cat(combined)
        shapes = [tuple(p.shape) for p in theta]

        # grad to x_query (direct dependence of the field on query coords).
        gxq_v = torch.autograd.grad(
            v, xq, grad_outputs=grad_v, retain_graph=True, allow_unused=True
        )[0]
        gxq_gv = torch.autograd.grad(
            gradv, xq, grad_outputs=grad_gradv, retain_graph=True, allow_unused=True
        )[0]
        gxq = (gxq_v if gxq_v is not None else 0) + (gxq_gv if gxq_gv is not None else 0)

        # HVP operator: H·u = ∇_θ ⟨g, u⟩, g = stationary gradient (graph to s kept).
        g = ctx.g_final
        shapes = [tuple(p.shape) for p in theta]

        def hvp(u):
            blocks = _flat_blocks(u, shapes)
            prod = sum((gi * ui).sum() for gi, ui in zip(g, blocks))
            hus = torch.autograd.grad(prod, theta, retain_graph=True, allow_unused=True)
            out = []
            for j, _ in enumerate(theta):
                h = hus[j] if j < len(hus) else None
                out.append(h.reshape(-1) if h is not None else torch.zeros_like(blocks[j]))
            return torch.cat(out)

        u = cg_solve(hvp, b_flat)
        if not torch.isfinite(u).all():
            u = torch.zeros_like(u)

        # grad to s: dL/ds = -∇_s ⟨g, u⟩.
        u_blocks = _flat_blocks(u, shapes)
        gu = sum((gi * ui).sum() for gi, ui in zip(g, u_blocks))
        (gs,) = torch.autograd.grad(gu, s, retain_graph=False, allow_unused=True)
        gs = -gs if gs is not None else torch.zeros_like(s)
        if not torch.isfinite(gs).all():
            gs = torch.zeros_like(s)

        grad_x = gxq if torch.is_tensor(gxq) else torch.zeros_like(x_query)
        if not torch.isfinite(grad_x).all():
            grad_x = torch.zeros_like(x_query)
        return grad_x, None, gs, None, None


class VariationalEstimator(nn.Module):
    """Variational (Ritz) Poisson estimator — online inner GD, bilevel (BOP) unless disabled."""

    def __init__(self, cfg: VariationalConfig, d: int):
        super().__init__()
        self.cfg = cfg
        self.v = PotentialHead(d, cfg.h, cfg.layers)
        self.n_bailouts = 0

    def forward(self, x_query: torch.Tensor, x_land: torch.Tensor, g_land: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the approximately-optimal v_θ* at query points.

        When ``cfg.bilevel`` is True the source ``g_land`` is kept in the graph
        and the outer loss differentiates through the inner minimizer back into
        the encoder (proper bilevel, PLAN §A.3 relaxed). When False it falls back
        to the original stop-gradient convention (field ignores the encoder;
        BC has no model path — known inert, kept for comparison).
        """
        if not self.cfg.bilevel:
            v, gradv = self._forward_inert(x_query, x_land, g_land)
        else:
            v, gradv = BOPPoissonSolve.apply(x_query, x_land, g_land, self.v, self.cfg)
        # Bailout guard: a divergent inner solve must never poison the outer
        # loss (0 * NaN = NaN kills even the lam=0 runs). On failure return a
        # zero field (regularizer contributes 0 this step) and re-init the head
        # so the warm-started inner solve starts fresh next step.
        if not (torch.isfinite(v).all() and torch.isfinite(gradv).all()).item():
            self.n_bailouts += 1
            self.v.reset_parameters()
            return (torch.zeros_like(v.detach()), torch.zeros_like(gradv.detach()))
        return v, gradv

    def _forward_inert(self, x_query, x_land, g_land):
        """Original stop-gradient path (deprecated; kept for the ablation)."""
        cfg = self.cfg
        x_bd = x_query.detach() if cfg.boundary_as_ood else None
        s = g_land.detach()
        opt = torch.optim.SGD(self.v.parameters(), lr=cfg.inner_lr)
        self.v.train()
        for _ in range(cfg.inner_steps):
            opt.zero_grad()
            loss = ritz_energy(self.v, x_land.detach(), s, x_bd,
                               mu=cfg.mu, lam_d=cfg.lam_d)
            loss.backward()
            opt.step()
        self.v.eval()
        v = self.v(x_query)
        gradv = self.v.gradv(x_query)
        return v, gradv
