import torch
import torch.nn as nn
import math
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
    mode: str = "ddpm"          # "ddpm", "gaussian", "shift_scale", "mixture", "rotate", "zoom", "mask", "time_shift", "fgsm"
    T: int = 200
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    sigma: float = 0.1
    shift_std: float = 0.2
    scale_std: float = 0.15
    # rotation (2D vectors only)
    rot_max_deg: float = 30.0
    # zoom / scaling
    zoom_min: float = 0.9
    zoom_max: float = 1.1
    # masking (feature/pixel dropout)
    mask_prob: float = 0.1
    mask_value: float = 0.0
    # time shift (for sequences: (B,T) or (B,T,D) or flattened (B,T) )
    time_shift_max: int = 5
    # fgsm
    fgsm_eps: float = 0.05
    p_ddpm: float = 0.5
    p_gaussian: float = 0.3
    p_shift_scale: float = 0.2


def fgsm_perturb(
    model: nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    eps: float,
    task: str,
) -> torch.Tensor:
    """Fast Gradient Sign Method perturbation.

    Uses the *downstream* loss:
      - reconstruction: MSE(model(x), y_true)
      - classification: CE(model(x), y_true)
      - regression: MSE(model(x), y_true)
    Returns x_adv = x + eps * sign(∇_x loss)
    """
    x_adv = x.detach().clone().requires_grad_(True)
    y_pred = model(x_adv)

    if task == "classification":
        loss = nn.CrossEntropyLoss()(y_pred, y_true)
    elif task in ("reconstruction", "regression"):
        loss = torch.mean((y_pred - y_true) ** 2)
    else:
        raise ValueError(f"Unknown task for FGSM: {task}")

    grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
    x_adv = (x_adv + eps * grad.sign()).detach()
    return x_adv

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

    def rotate_corrupt(self, x: torch.Tensor) -> torch.Tensor:
        """Random 2D rotation for vector inputs.

        Supports x shaped (B,2). For other shapes, raises.
        """
        if x.dim() != 2 or x.size(1) != 2:
            raise ValueError("rotate corruption currently supports only x with shape (B, 2)")
        B = x.size(0)
        theta = (2.0 * torch.rand(B, device=x.device, dtype=x.dtype) - 1.0) * (self.cfg.rot_max_deg * math.pi / 180.0)
        c = torch.cos(theta)
        s = torch.sin(theta)
        R = torch.stack([
            torch.stack([c, -s], dim=1),
            torch.stack([s,  c], dim=1)
        ], dim=1)  # (B,2,2)
        return torch.bmm(R, x.unsqueeze(-1)).squeeze(-1)

    def zoom_corrupt(self, x: torch.Tensor) -> torch.Tensor:
        """Random zoom / scaling.

        - For vector inputs (B,d): scales each sample by s ~ U[zoom_min, zoom_max]
        - For images (B,C,H,W): same per-sample scalar.
        """
        B = x.size(0)
        s = torch.empty(B, device=x.device, dtype=x.dtype).uniform_(self.cfg.zoom_min, self.cfg.zoom_max)
        view = [B] + [1] * (x.dim() - 1)
        return x * s.view(*view)

    def mask_corrupt(self, x: torch.Tensor) -> torch.Tensor:
        """Random masking.

        Low-dimensional unstructured data: zeros random coordinates.
        Images: zeros random pixels (independent Bernoulli mask).
        """
        if self.cfg.mask_prob <= 0:
            return x
        # Bernoulli keep mask
        keep = (torch.rand_like(x) > self.cfg.mask_prob).to(x.dtype)
        return x * keep + (1.0 - keep) * torch.tensor(self.cfg.mask_value, device=x.device, dtype=x.dtype)

    def time_shift_corrupt(self, x: torch.Tensor) -> torch.Tensor:
        """Random time shift.

        Supports:
          - (B, T) : roll along dim=1
          - (B, T, D) : roll along dim=1
        For other shapes, raises.
        """
        if x.dim() not in (2, 3):
            raise ValueError("time_shift corruption expects x with shape (B,T) or (B,T,D)")
        B = x.size(0)
        kmax = int(self.cfg.time_shift_max)
        if kmax <= 0:
            return x
        shifts = torch.randint(-kmax, kmax + 1, (B,), device=x.device)
        x_out = x.clone()
        # x[i] has shape (T) or (T,D) so rolling along dim=0 is correct.
        for i in range(B):
            x_out[i] = torch.roll(x[i], shifts=int(shifts[i].item()), dims=0)
        return x_out

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mode = self.cfg.mode
        if mode == "ddpm":
            x_t, t_used = self.ddpm_corrupt(x, t=t)
            return x_t, t_used
        if mode == "gaussian":
            return self.gaussian_corrupt(x), None
        if mode == "shift_scale":
            return self.shift_scale_corrupt(x), None
        if mode == "rotate":
            return self.rotate_corrupt(x), None
        if mode == "zoom":
            return self.zoom_corrupt(x), None
        if mode == "mask":
            return self.mask_corrupt(x), None
        if mode == "time_shift":
            return self.time_shift_corrupt(x), None
        if mode == "fgsm":
            # FGSM needs access to the model and task loss; call fgsm_perturb(...) from the training loop.
            return x, None
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