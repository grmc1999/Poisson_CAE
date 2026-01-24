import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


import argparse
import torchvision
import torchvision.transforms as T

from Utils.grad_operations import *
from Utils.projectors import CorruptionOperator
from Utils.projectors import CorruptionConfig
from Utils.geometry_estimators import PoissonMCConfig
from Utils.geometry_estimators import PoissonMCEstimator
from Utils.visualization import visualize_fields_2d, Viz2DConfig, visualize_mnist_fields, VizImageConfig

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
    viz_every: int = 500,
    viz_dir: str = "outputs",
):
    model.to(device).train()
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
            v, grad_v = PR.Estimate_field_grads(x, x_tilde, landmarks=landmarks)

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

            # Visualize learned fields (2D toy case)
            if viz_every > 0 and (step % viz_every == 0):
                try:
                    _ = visualize_fields_2d(
                        model=model,
                        poisson_reg=PR,
                        projector=Pi,
                        x_batch=x.detach(),
                        out_dir=viz_dir,
                        step=step,
                        device=device,
                        cfg=Viz2DConfig(grid_n=160, padding=0.75, landmarks=landmarks, dpi=160),
                    )
                except Exception as e:
                    print(f"[viz] warning: visualization failed at step {step}: {e}")
            if step >= steps:
                return x_hat



def run_mnist(
    *,
    device: str,
    steps: int = 2000,
    batch_size: int = 128,
    sigma: float = 0.3,
    lam: float = 1e-2,
    landmarks: int = 64,
    viz_every: int = 500,
    viz_dir: str = "outputs",
    z_dim: int = 64,
    hidden: int = 512,
):
    # MNIST -> flatten to vectors in [0,1]
    tfm = T.Compose([T.ToTensor()])
    ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    d = 28 * 28

    # Encoder/Decoder as MLP on flattened images (consistent with repo style)
    enc = Encoder(d=d, h=hidden, z=z_dim)
    dec = Decoder(z=z_dim, h=hidden, d=d)
    model = AE_model(enc, dec)

    # Gaussian corruption in pixel space
    Pi = CorruptionOperator(CorruptionConfig(mode="gaussian", sigma=sigma))

    # Poisson estimator in input space (d=784 is high; keep landmarks small)
    poisson_est = PoissonMCEstimator(PoissonMCConfig(eps=1e-2, landmarks=landmarks))
    model.to(device)

    # Train loop (reuse existing train(), but adapt data: (x,) -> flatten)
    model.to(device).train()
    Pi.to(device).train()
    poisson_est.to(device).train()

    opt = torch.optim.Adam(list(model.parameters()), lr=1e-3)
    PR = Poisson_reg(poisson_est, model)

    step = 0
    for epoch in range(10**9):
        for x_img, y in loader:
            x_img = x_img.to(device)
            x = x_img.view(x_img.size(0), -1).requires_grad_(True)  # (B,784)

            x_tilde, _ = Pi(x)

            x_hat = model.forward(x_tilde)

            v, grad_v = PR.Estimate_field_grads(x, x_tilde, landmarks=landmarks)
            logp = PR.ML_loss(x, x_hat)
            flux = PR.BC_loss(x, x_tilde, grad_v)
            bulk = PR.D_loss(x, x_hat, grad_v)

            loss = logp + lam * (flux + bulk)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            step += 1
            if step % 100 == 0:
                print(f"[mnist] step={step} recon={logp.item():.6f} flux={flux.item():.6f} bulk={bulk.item():.6f} loss={loss.item():.6f}")

            if viz_every > 0 and (step % viz_every == 0):
                try:
                    # visualize using the same PR + Pi; pass labels for the chosen subset
                    _ = visualize_mnist_fields(
                        model=model,
                        poisson_reg=PR,
                        projector=Pi,
                        x_batch_flat=x.detach(),
                        x_shape=(1, 28, 28),
                        labels=y.detach(),
                        out_dir=viz_dir,
                        step=step,
                        device=device,
                        cfg=VizImageConfig(num_images=8, landmarks=landmarks, dpi=160),
                    )
                except Exception as e:
                    print(f"[mnist-viz] warning: {e}")

            if step >= steps:
                return


# -----------------------------
# Minimal runnable toy experiment (2D)
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="toy", choices=["toy", "mnist"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--lam", type=float, default=1e-2)
    parser.add_argument("--landmarks", type=int, default=64)
    parser.add_argument("--viz_every", type=int, default=500)
    parser.add_argument("--viz_dir", type=str, default="outputs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "mnist":
        run_mnist(
            device=device,
            steps=args.steps,
            batch_size=args.batch_size,
            sigma=args.sigma,
            lam=args.lam,
            landmarks=args.landmarks,
            viz_every=args.viz_every,
            viz_dir=args.viz_dir,
        )
    else:
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
