import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


def affine_sample_2d(img: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Differentiable bilinear affine resampling of a single-channel image batch.

    Drop-in replacement for ``F.grid_sample(F.affine_grid(theta, img.shape),
    align_corners=False)`` with zero padding. ``theta`` is ``(B, 2, 3)`` and maps
    output normalized coordinates to input normalized coordinates.

    Implemented with ``gather`` + arithmetic instead of ``F.grid_sample`` because
    ``aten::grid_sampler_2d_backward`` has no double-backward implementation,
    which the bilevel Poisson estimator requires. Since the sampling grid does not
    depend on ``img`` values, the operation is linear in ``img`` and therefore
    supports arbitrary higher-order derivatives.
    """
    B, C, H, W = img.shape
    device, dtype = img.device, img.dtype

    ys = torch.arange(H, device=device, dtype=dtype)
    xs = torch.arange(W, device=device, dtype=dtype)
    # normalized coords, align_corners=False: (2*idx + 1)/size - 1
    gy = (2.0 * ys + 1.0) / H - 1.0
    gx = (2.0 * xs + 1.0) / W - 1.0
    gyy, gxx = torch.meshgrid(gy, gx, indexing="ij")
    ones = torch.ones_like(gxx)
    coords = torch.stack([gxx, gyy, ones], dim=-1).reshape(-1, 3)  # (H*W, 3)

    inorm = torch.einsum("bij,nj->bni", theta, coords)  # (B, H*W, 2)
    # back to pixel coordinates: pix = ((norm + 1)*size - 1)/2
    ix = ((inorm[..., 0] + 1.0) * W - 1.0) / 2.0
    iy = ((inorm[..., 1] + 1.0) * H - 1.0) / 2.0

    x0 = torch.floor(ix)
    y0 = torch.floor(iy)
    dx = (ix - x0).unsqueeze(1)  # (B, 1, H*W)
    dy = (iy - y0).unsqueeze(1)
    x0 = x0.long()
    y0 = y0.long()

    flat = img.reshape(B, C, H * W)

    def _gather(yy, xx):
        valid = ((yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)).unsqueeze(1).to(dtype)
        idx = (yy.clamp(0, H - 1) * W + xx.clamp(0, W - 1)).unsqueeze(1)
        return flat.gather(2, idx.expand(-1, C, -1)) * valid

    w00 = (1.0 - dx) * (1.0 - dy)
    w01 = dx * (1.0 - dy)
    w10 = (1.0 - dx) * dy
    w11 = dx * dy
    out = (
        w00 * _gather(y0, x0)
        + w01 * _gather(y0, x0 + 1)
        + w10 * _gather(y0 + 1, x0)
        + w11 * _gather(y0 + 1, x0 + 1)
    )
    return out.reshape(B, C, H, W)

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
    mode: str = "ddpm"          # "ddpm", "gaussian", "shift_scale", "mixture", "mask", "dropout",
                                # "rotation", "zoom"
    T: int = 200
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    sigma: float = 0.1
    shift_std: float = 0.2
    scale_std: float = 0.15
    p_ddpm: float = 0.5
    p_gaussian: float = 0.3
    p_shift_scale: float = 0.2
    mask_frac: float = 0.3       # fraction of dims zeroed (mask mode)
    drop_p: float = 0.2          # probability of dropping a dim (dropout mode)
    rotation_max_deg: float = 30.0  # max |angle| (rotation mode)
    zoom_std: float = 0.15       # max |scale-1| (zoom mode)
    image_side: int = 0          # >0 -> interpret flat input as (side, side) images

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

    def mask_corrupt(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        frac = self.cfg.mask_frac
        mask = (torch.rand(B, D, device=x.device) > frac).float()
        return x * mask

    def dropout_corrupt(self, x: torch.Tensor) -> torch.Tensor:
        keep = 1.0 - self.cfg.drop_p
        mask = (torch.rand_like(x) < keep).float()
        return x * mask

    def _rotation_matrices(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample 2x2 rotation matrices, angle ~ U(-max_deg, max_deg)."""
        B = x.size(0)
        if self.cfg.rotation_max_deg <= 0:
            return torch.eye(2, device=x.device, dtype=x.dtype).expand(B, 2, 2).clone()
        ang = (2.0 * torch.rand(B, device=x.device, dtype=x.dtype) - 1.0) * (
            self.cfg.rotation_max_deg * math.pi / 180.0
        )
        cos, sin = torch.cos(ang), torch.sin(ang)
        row0 = torch.stack([cos, -sin], dim=1)
        row1 = torch.stack([sin, cos], dim=1)
        return torch.stack([row0, row1], dim=1)  # (B,2,2)

    def rotation_corrupt(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        if self.cfg.rotation_max_deg <= 0:
            return x
        R = self._rotation_matrices(x)
        if self.cfg.image_side > 0:
            s = self.cfg.image_side
            img = x.view(B, 1, s, s)
            theta = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
            theta[:, :2, :2] = R
            return affine_sample_2d(img, theta).view(B, -1)
        if x.size(1) < 2:
            return x
        # rotate the first two coordinates about the origin
        xy = x[:, :2]
        xy_rot = torch.bmm(xy.unsqueeze(1), R.transpose(1, 2)).squeeze(1)
        if x.size(1) == 2:
            return xy_rot
        return torch.cat([xy_rot, x[:, 2:]], dim=1)

    def zoom_corrupt(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        if self.cfg.zoom_std <= 0:
            return x
        scale = 1.0 + (2.0 * torch.rand(B, device=x.device, dtype=x.dtype) - 1.0) * self.cfg.zoom_std
        if self.cfg.image_side > 0:
            s = self.cfg.image_side
            img = x.view(B, 1, s, s)
            theta = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
            # sampling grid shrinks for scale>1 -> magnify (zoom in)
            theta[:, 0, 0] = 1.0 / scale
            theta[:, 1, 1] = 1.0 / scale
            return affine_sample_2d(img, theta).view(B, -1)
        return x * scale.view(B, *([1] * (x.dim() - 1)))

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mode = self.cfg.mode
        if mode == "ddpm":
            x_t, t_used = self.ddpm_corrupt(x, t=t)
            return x_t, t_used
        if mode == "gaussian":
            return self.gaussian_corrupt(x), None
        if mode == "shift_scale":
            return self.shift_scale_corrupt(x), None
        if mode == "mask":
            return self.mask_corrupt(x), None
        if mode == "dropout":
            return self.dropout_corrupt(x), None
        if mode == "rotation":
            return self.rotation_corrupt(x), None
        if mode == "zoom":
            return self.zoom_corrupt(x), None
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