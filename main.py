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
            x = x.to(device).requires_grad_(True)

            # Corrupt / OOD via Πψ
            x_tilde, _ = Pi(x)

            x_hat = model.forward(x)
            v, grad_v = PR.Estimate_field_grads(x,x_tilde,landmarks = 100)

            # CAE reconstruction: reconstruct clean x from corrupted hidden state
            logp = PR.ML_loss(x,x_hat)

            flux = PR.BC_loss(x,x_tilde,grad_v)

            bulk = PR.D_loss(x,x_hat,grad_v)

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
