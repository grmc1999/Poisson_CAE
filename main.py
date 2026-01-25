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
from Utils.visualization import visualize_fields_2d, Viz2DConfig

from models import Poisson_reg, AE_model, Classifier_model, Regressor_model, GRUEncoder
from Utils.datasets import get_experiment_loaders, LoaderCfg



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
        for batch in dataloader:
            if len(batch) == 1:
                x = batch[0]
                y_true = x  # reconstruction
            else:
                x, y_true = batch
            x = x.to(device).requires_grad_(True)
            y_true = y_true.to(device)

            # Corrupt / OOD via Πψ
            x_tilde, _ = Pi(x)

            # Downstream prediction from corrupted input (CAE-style)
            y_pred = model.forward(x_tilde)
            print(y_pred,y_true)
            v, grad_v = PR.Estimate_field_grads(x, x_tilde, landmarks=landmarks)

            logp = PR.ML_loss(y_true, y_pred)
            flux = PR.BC_loss(x, x_tilde, grad_v)
            bulk = PR.D_loss(x, y_true, y_pred, grad_v)

            loss = logp + lam * (flux + bulk)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            step += 1
            if step % 200 == 0:
                print(
                    f"step={step} recon={logp.item():.6f} "
                    f"flux={flux.item():.6f} bulk={bulk.item():.6f} loss={loss.item():.6f}"
                )

            # Visualize learned fields (2D toy case)
            if viz_every > 0 and (step % viz_every == 0):
                # Only meaningful for 2D inputs
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
                return


# -----------------------------
# Minimal runnable toy experiment (2D)
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import argparse

    parser = argparse.ArgumentParser(description='Poisson-CAE experiments')
    parser.add_argument('--experiment', type=str, default='mog',
                        choices=['mog','spirals','banana','rings','breast_cancer','sinusoid_reg'],
                        help='Experiment to run')
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--lam', type=float, required=True)
    parser.add_argument('--landmarks', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--viz_every', type=int, default=500)
    parser.add_argument('--viz_dir', type=str, default='outputs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--encoder_type', type=str, default='mlp', choices=['mlp','gru'],
                        help='Encoder type (use gru for sinusoid_reg time-series)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Data loaders
    if args.experiment == 'mog':
        # Legacy: mixture of Gaussians reconstruction in R^2
        N = 5000
        centers = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.25]], dtype=torch.float32)
        comp = torch.randint(0, centers.size(0), (N,))
        x = centers[comp] + 0.15 * torch.randn(N, 2)
        loader = DataLoader(TensorDataset(x), batch_size=args.batch, shuffle=True, drop_last=True)
        test_loader = None
        input_dim = 2
        task = 'reconstruction'
    else:
        loader, test_loader, input_dim, task = get_experiment_loaders(
            args.experiment,
            LoaderCfg(batch_size=args.batch, shuffle=True, drop_last=True),
            seed=args.seed,
        )

    # Corruption operator Πψ
    Pi = CorruptionOperator(CorruptionConfig(mode="gaussian", T=200, beta_start=1e-4, beta_end=2e-2))

    # Poisson Monte Carlo estimator
    poisson_est = PoissonMCEstimator(PoissonMCConfig(eps=1e-2, landmarks=args.landmarks))

    # Model selection by task
    if task == 'reconstruction':
        model = AE_model(
            Encoder(d=input_dim, h=128, z=32),
            Decoder(z=32, h=128, d=input_dim)
        )
    elif task == 'classification':
        # all classification tasks here are binary (2 classes)
        model = Classifier_model(Encoder(d=input_dim, h=128, z=32), n_classes=2)
    elif task == 'regression':
        # sinusoid_reg: regress (A, f, phi)
        if args.encoder_type == 'gru':
            # input is flattened sequence of length input_dim, assume 1D signal
            enc = GRUEncoder(T=input_dim, din=1, hidden=64, z_dim=32)
        else:
            enc = Encoder(d=input_dim, h=128, z=32)
        model = Regressor_model(enc, out_dim=3)
    else:
        raise ValueError(f"Unknown task: {task}")

    train(
        model = model,
        Pi=Pi,
        poisson_est=poisson_est,
        dataloader=loader,
        lr=args.lr,
        lam=args.lam,
        landmarks=args.landmarks,
        device=device,
        steps=args.steps,
        viz_every=args.viz_every,
        viz_dir=args.viz_dir,
    )

    # Quick test evaluation when available
    if test_loader is not None:
        model.eval()
        correct = 0
        total = 0
        mse = 0.0
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                if task == 'classification':
                    pred = y_pred.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.numel()
                elif task == 'regression':
                    mse += ((y_pred - y) ** 2).mean().item()
                    total += 1
        if task == 'classification':
            print(f"[test] accuracy={correct/float(total):.4f} ({correct}/{total})")
        elif task == 'regression':
            print(f"[test] mse={mse/float(total):.6f}")

