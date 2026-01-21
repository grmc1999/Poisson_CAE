import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)

def make_ddpm_coeffs(betas: torch.Tensor) -> Dict[str, torch.Tensor]:
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": torch.sqrt(alpha_bars),
        "sqrt_one_minus_alpha_bars": torch.sqrt(1.0 - alpha_bars),
    }

@dataclass
class CorruptionConfig:
    mode: str = "ddpm"          # "ddpm", "gaussian", "shift_scale", "mixture"
    T: int = 200
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    sigma: float = 0.1
    shift_std: float = 0.2
    scale_std: float = 0.15
    p_ddpm: float = 0.5
    p_gaussian: float = 0.3
    p_shift_scale: float = 0.2

class CorruptionOperator(nn.Module):
    def __init__(self, cfg: CorruptionConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.mode in ("ddpm", "mixture"):
            betas = linear_beta_schedule(cfg.T, cfg.beta_start, cfg.beta_end)
            coeffs = make_ddpm_coeffs(betas)
            for k, v in coeffs.items():
                self.register_buffer(k, v)

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(low=0, high=self.cfg.T, size=(batch_size,), device=device)

    def ddpm_corrupt(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        device = x.device
        if t is None:
            t = self.sample_timesteps(B, device=device)

        eps = torch.randn_like(x)
        s1 = self.sqrt_alpha_bars[t].view(B, *([1] * (x.dim() - 1)))
        s2 = self.sqrt_one_minus_alpha_bars[t].view(B, *([1] * (x.dim() - 1)))
        x_t = s1 * x + s2 * eps
        return x_t, t

    def gaussian_corrupt(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cfg.sigma * torch.randn_like(x)

    def shift_scale_corrupt(self, x: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.cfg.scale_std * torch.randn(x.size(0), 1, device=x.device, dtype=x.dtype)
        shift = self.cfg.shift_std * torch.randn_like(x)
        return scale * x + shift

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mode = self.cfg.mode
        if mode == "ddpm":
            x_t, t_used = self.ddpm_corrupt(x, t=t)
            return x_t, t_used
        if mode == "gaussian":
            return self.gaussian_corrupt(x), None
        if mode == "shift_scale":
            return self.shift_scale_corrupt(x), None
        if mode == "mixture":
            B = x.size(0)
            device = x.device
            probs = torch.tensor([self.cfg.p_ddpm, self.cfg.p_gaussian, self.cfg.p_shift_scale], device=device)
            probs = probs / probs.sum()
            choices = torch.multinomial(probs, num_samples=B, replacement=True)

            x_out = x.clone()
            t_used = torch.full((B,), -1, device=device, dtype=torch.long)

            idx0 = (choices == 0).nonzero(as_tuple=False).squeeze(1)
            if idx0.numel() > 0:
                x_ddpm, t0 = self.ddpm_corrupt(x[idx0], t=None)
                x_out[idx0] = x_ddpm
                t_used[idx0] = t0

            idx1 = (choices == 1).nonzero(as_tuple=False).squeeze(1)
            if idx1.numel() > 0:
                x_out[idx1] = self.gaussian_corrupt(x[idx1])

            idx2 = (choices == 2).nonzero(as_tuple=False).squeeze(1)
            if idx2.numel() > 0:
                x_out[idx2] = self.shift_scale_corrupt(x[idx2])

            return x_out, t_used
        raise ValueError(f"Unknown corruption mode: {mode}")