import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Utils.grad_operations import *
from Utils.projectors import CorruptionOperator
from Utils.projectors import CorruptionConfig
from Utils.geometry_estimators import PoissonMCConfig
from Utils.geometry_estimators import PoissonMCEstimator
from models import Poisson_reg,AE_model


# -----------------------------
# Corruption operator Πψ (from earlier)
# -----------------------------
#def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
#    return torch.linspace(beta_start, beta_end, T)
#
#def make_ddpm_coeffs(betas: torch.Tensor) -> Dict[str, torch.Tensor]:
#    alphas = 1.0 - betas
#    alpha_bars = torch.cumprod(alphas, dim=0)
#    return {
#        "betas": betas,
#        "alphas": alphas,
#        "alpha_bars": alpha_bars,
#        "sqrt_alpha_bars": torch.sqrt(alpha_bars),
#        "sqrt_one_minus_alpha_bars": torch.sqrt(1.0 - alpha_bars),
#    }
#
#@dataclass
#class CorruptionConfig:
#    mode: str = "ddpm"          # "ddpm", "gaussian", "shift_scale", "mixture"
#    T: int = 200
#    beta_start: float = 1e-4
#    beta_end: float = 2e-2
#    sigma: float = 0.1
#    shift_std: float = 0.2
#    scale_std: float = 0.15
#    p_ddpm: float = 0.5
#    p_gaussian: float = 0.3
#    p_shift_scale: float = 0.2
#
#class CorruptionOperator(nn.Module):
#    def __init__(self, cfg: CorruptionConfig):
#        super().__init__()
#        self.cfg = cfg
#        if cfg.mode in ("ddpm", "mixture"):
#            betas = linear_beta_schedule(cfg.T, cfg.beta_start, cfg.beta_end)
#            coeffs = make_ddpm_coeffs(betas)
#            for k, v in coeffs.items():
#                self.register_buffer(k, v)
#
#    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
#        return torch.randint(low=0, high=self.cfg.T, size=(batch_size,), device=device)
#
#    def ddpm_corrupt(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#        B = x.size(0)
#        device = x.device
#        if t is None:
#            t = self.sample_timesteps(B, device=device)
#
#        eps = torch.randn_like(x)
#        s1 = self.sqrt_alpha_bars[t].view(B, *([1] * (x.dim() - 1)))
#        s2 = self.sqrt_one_minus_alpha_bars[t].view(B, *([1] * (x.dim() - 1)))
#        x_t = s1 * x + s2 * eps
#        return x_t, t
#
#    def gaussian_corrupt(self, x: torch.Tensor) -> torch.Tensor:
#        return x + self.cfg.sigma * torch.randn_like(x)
#
#    def shift_scale_corrupt(self, x: torch.Tensor) -> torch.Tensor:
#        scale = 1.0 + self.cfg.scale_std * torch.randn(x.size(0), 1, device=x.device, dtype=x.dtype)
#        shift = self.cfg.shift_std * torch.randn_like(x)
#        return scale * x + shift
#
#    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#        mode = self.cfg.mode
#        if mode == "ddpm":
#            x_t, t_used = self.ddpm_corrupt(x, t=t)
#            return x_t, t_used
#        if mode == "gaussian":
#            return self.gaussian_corrupt(x), None
#        if mode == "shift_scale":
#            return self.shift_scale_corrupt(x), None
#        if mode == "mixture":
#            B = x.size(0)
#            device = x.device
#            probs = torch.tensor([self.cfg.p_ddpm, self.cfg.p_gaussian, self.cfg.p_shift_scale], device=device)
#            probs = probs / probs.sum()
#            choices = torch.multinomial(probs, num_samples=B, replacement=True)
#
#            x_out = x.clone()
#            t_used = torch.full((B,), -1, device=device, dtype=torch.long)
#
#            idx0 = (choices == 0).nonzero(as_tuple=False).squeeze(1)
#            if idx0.numel() > 0:
#                x_ddpm, t0 = self.ddpm_corrupt(x[idx0], t=None)
#                x_out[idx0] = x_ddpm
#                t_used[idx0] = t0
#
#            idx1 = (choices == 1).nonzero(as_tuple=False).squeeze(1)
#            if idx1.numel() > 0:
#                x_out[idx1] = self.gaussian_corrupt(x[idx1])
#
#            idx2 = (choices == 2).nonzero(as_tuple=False).squeeze(1)
#            if idx2.numel() > 0:
#                x_out[idx2] = self.shift_scale_corrupt(x[idx2])
#
#            return x_out, t_used
#        raise ValueError(f"Unknown corruption mode: {mode}")
#
#
# -----------------------------
# CAE: encoder f(x) and decoder
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, d: int, h: int = 128, z: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, z),
        )
    def forward(self, x):
        return self.net(x)  # f(x)

class Decoder(nn.Module):
    def __init__(self, z: int = 32, h: int = 128, d: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, d),
        )
    def forward(self, z):
        return self.net(z)
#
#
#
## -----------------------------
## Regularized Green kernel and its gradient
## (free-space Laplacian Green's function; good for toy experiments)
# -----------------------------
#def omega_d(d: int) -> float:
#    return 2.0 * (math.pi ** (d / 2.0)) / math.gamma(d / 2.0)
#
#def green_reg(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
#    """
#    x: (B, d), y: (M, d) => G: (B, M)
#    """
#    B, d = x.shape
#    r2 = ((x[:, None, :] - y[None, :, :]) ** 2).sum(dim=2)
#    r = torch.sqrt(r2 + eps**2)
#    if d == 2:
#        return -(1.0 / (2.0 * math.pi)) * torch.log(r)
#    if d >= 3:
#        c = 1.0 / ((d - 2.0) * omega_d(d))
#        return c * (r ** (2.0 - d))
#    raise ValueError("d must be >=2")
#
#def gradx_green_reg(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
#    """
#    ∇_x G(x,y): returns (B, M, d)
#    """
#    B, d = x.shape
#    diff = x[:, None, :] - y[None, :, :]
#    r2 = (diff ** 2).sum(dim=2)
#    r = torch.sqrt(r2 + eps**2)
#    if d == 2:
#        # G = -(1/(2π)) log r  => ∇ = -(1/(2π)) diff / (r^2)
#        return -(1.0 / (2.0 * math.pi)) * diff / (r2[..., None] + eps**2)
#    if d >= 3:
#        c = 1.0 / ((d - 2.0) * omega_d(d))
#        return c * (2.0 - d) * diff * (r[..., None] ** (-d))
#    raise ValueError("d must be >=2")
#
#
## -----------------------------
## Poisson potential estimator (Monte Carlo Green integral)
## v(x) ≈ (1/M) Σ_j G(x, y_j) g(y_j)
## ∇v(x) ≈ (1/M) Σ_j ∇_x G(x, y_j) g(y_j)
## -----------------------------
#@dataclass
#class PoissonMCConfig:
#    eps: float = 1e-2
#    landmarks: int = 256          # M (per batch)
#    exclude_self: bool = False    # for simplicity in minibatch setting
#
#class PoissonMCEstimator(nn.Module):
#    def __init__(self, cfg: PoissonMCConfig):
#        super().__init__()
#        self.cfg = cfg
#
#    def forward(self, x_query: torch.Tensor, x_land: torch.Tensor, g_land: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#        """
#        Returns:
#          v_hat: (B,)
#          gradv_hat: (B, d)
#        All differentiable w.r.t x_query and g_land.
#        """
#        B, d = x_query.shape
#        M = x_land.shape[0]
#        denom = float(M)
#
#        G = green_reg(x_query, x_land, eps=self.cfg.eps)              # (B, M)
#        dG = gradx_green_reg(x_query, x_land, eps=self.cfg.eps)       # (B, M, d)
#
#        v_hat = (G * g_land[None, :]).sum(dim=1) / denom              # (B,)
#        gradv_hat = (dG * g_land[None, :, None]).sum(dim=1) / denom   # (B, d)
#        return v_hat, gradv_hat

#
# -----------------------------
# Regularizer term: E[ ∇v(Πψ(x)) · nψ(x) ]
# with nψ defined from the SAME Πψ as a direction of corruption
# -----------------------------
#def boundary_like_regularizer(
#    encoder: nn.Module,
#    poisson_est: PoissonMCEstimator,
#    x_clean: torch.Tensor,
#    x_tilde: torch.Tensor,
#    landmarks: int = 256,
#) -> torch.Tensor:
#    """
#    x_clean: (B, d)
#    x_tilde: (B, d) = Πψ(x_clean)
#    returns scalar reg
#    """
#
#    B, d = x_clean.shape
#
#    # Normal induced by corruption map Πψ:
#    # nψ(x) = (Πψ(x) - x) / ||Πψ(x) - x||
#    delta = x_tilde - x_clean
#    n = delta / (delta.norm(dim=1, keepdim=True) + 1e-8)  # (B, d)
#
#    # Choose landmark points from the corrupted batch (Monte Carlo quadrature set)
#    M = min(landmarks, B)
#    idx = torch.randperm(B, device=x_clean.device)[:M]
#    x_land = x_tilde[idx].clone().requires_grad_(True)  # (M, d)
#
#    # Source term g(y)=||∇f(y)||_F at landmarks; requires higher-order grads for training
#    g_land = jacobian_fro_norm(encoder, x_land, create_graph=True)   # (M,)
#
#    # Estimate ∇v at query points x_tilde (the corrupted/OOD points)
#    x_query = x_tilde  # (B, d)
#    _, gradv = poisson_est(x_query, x_land, g_land)  # gradv: (B, d)
#
#    # Boundary-like flux term
#    reg = (gradv * n).sum(dim=1).mean()
#    return reg


# -----------------------------
# Training loop
# -----------------------------
def train(
    model: nn.Module,
    Pi: CorruptionOperator,
    poisson_est: PoissonMCEstimator,
    dataloader: DataLoader,
    *,
    lr: float = 1e-3,
    lam: float = 1e-2,
    landmarks: int = 256,
    device: str = "cuda",
    steps: int = 5000,
):
    model.to(device).train()
    decoder.to(device).train()
    Pi.to(device).train()
    poisson_est.to(device).train()

    opt = torch.optim.Adam(list(model.parameters()), lr=lr)

    PR = Poisson_reg(poisson_est, model)

    step = 0
    for epoch in range(10**9):
        for (x,) in dataloader:
            x = x.to(device)

            # Corrupt / OOD via Πψ
            x_tilde, _ = Pi(x)

            x_hat = model.forward(x)
            v, grad_v = PR.Estimate_field_grads(x,x_tilde,landmarks = 100)

            # CAE reconstruction: reconstruct clean x from corrupted hidden state
            logp = PR.ML_loss(x,x_hat)

            flux = PR.BC_loss(x,x_tilde,grad_v)

            bulk = PR.D_loss(x,grad_v)

            loss = logp + lam * (flux + bulk)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            step += 1
            if step % 200 == 0:
                print(f"step={step}\
                       recon={logp.item():.6f}\
                      flux={flux.item():.6f}\
                      bulk={bulk.item():.6f}\
                      loss={loss.item():.6f}")
            if step >= steps:
                return x_hat


# -----------------------------
# Minimal runnable toy experiment (2D)
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Toy in-domain data: mixture of Gaussians in R^2
    torch.manual_seed(0)
    N = 5000
    centers = torch.tensor([[ -1.0, 0.0 ],
                            [  1.0, 0.0 ],
                            [  0.0, 1.25 ]], dtype=torch.float32)
    comp = torch.randint(0, centers.size(0), (N,))
    x = centers[comp] + 0.15 * torch.randn(N, 2)

    loader = DataLoader(TensorDataset(x), batch_size=256, shuffle=True, drop_last=True)

    encoder = Encoder(d=2, h=128, z=32)
    decoder = Decoder(z=32, h=128, d=2)

    # Corruption operator Πψ: diffusion-like by default (your "go" -> ddpm)
    Pi = CorruptionOperator(CorruptionConfig(mode="ddpm", T=200, beta_start=1e-4, beta_end=2e-2))

    # Poisson Monte Carlo estimator
    poisson_est = PoissonMCEstimator(PoissonMCConfig(eps=1e-2, landmarks=256))

    #Poisson_reg
    model = AE_model(
        Encoder(d=2, h=128, z=32),
        Decoder(z=32, h=128, d=2)
    )

    h = train(
        model = model,
        Pi=Pi,
        poisson_est=poisson_est,
        dataloader=loader,
        lr=1e-3,
        lam=1e-2,
        landmarks=256,
        device=device,
        steps=1000,
    )