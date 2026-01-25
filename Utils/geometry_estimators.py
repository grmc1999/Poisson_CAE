import torch
import torch.nn as nn
from .grad_operations import green_reg, gradx_green_reg
from dataclasses import dataclass

@dataclass
class PoissonMCConfig:
    eps: float = 1e-2
    landmarks: int = 256          # M (per batch)
    exclude_self: bool = False    # for simplicity in minibatch setting

class PoissonMCEstimator(nn.Module):
    def __init__(self, cfg: PoissonMCConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, x_query: torch.Tensor, x_land: torch.Tensor, g_land: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          v_hat: (B,)
          gradv_hat: (B, d)
        All differentiable w.r.t x_query and g_land.
        """
        B, d = x_query.shape
        M = x_land.shape[0]
        denom = float(M)

        G = green_reg(x_query, x_land, eps=self.cfg.eps)              # (B, M)
        print("G",G)
        dG = gradx_green_reg(x_query, x_land, eps=self.cfg.eps)       # (B, M, d)
        print("dG",dG)

        v_hat = (G * g_land[None, :]).sum(dim=1) / denom              # (B,)
        gradv_hat = (dG * g_land[None, :, None]).sum(dim=1) / denom   # (B, d)
        return v_hat, gradv_hat