"""MNIST (flattened) experiment.

Runs the Poisson-regularized AE in *pixel space* (d=784) with Gaussian corruption.

Visualizations saved to ./outputs via Utils.visualization.visualize_mnist_flat
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from Utils.projectors import CorruptionOperator, CorruptionConfig
from Utils.geometry_estimators import PoissonMCConfig, PoissonMCEstimator
from Utils.grad_operations import jacobian_fro_norm
from Utils.visualization import visualize_mnist_flat, VizConfig

from models import AE_model, Poisson_reg, FlatEncoderMNIST, FlatDecoderMNIST


def main(
    *,
    batch_size: int = 128,
    steps: int = 3000,
    lr: float = 2e-4,
    lam: float = 1e-2,
    landmarks: int = 64,
    sigma: float = 0.3,
    viz_every: int = 500,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Data (keep [0,1])
    tfm = transforms.ToTensor()
    ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)

    # Model
    enc = FlatEncoderMNIST(d_in=784, d_hidden=512, z_dim=64)
    dec = FlatDecoderMNIST(z_dim=64, d_hidden=512, d_out=784)
    model = AE_model(enc, dec).to(device)

    # Corruption Πψ: Gaussian noise
    Pi = CorruptionOperator(CorruptionConfig(mode="gaussian", sigma=sigma)).to(device)

    # Poisson estimator in pixel space
    poisson_est = PoissonMCEstimator(PoissonMCConfig(eps=1e-2, landmarks=landmarks)).to(device)
    PR = Poisson_reg(poisson_est, model).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    viz_cfg = VizConfig(out_dir="outputs", max_items=8)

    step = 0
    for x, _ in dl:
        if step >= steps:
            break

        x = x.to(device)
        x = x.view(x.size(0), -1).requires_grad_(True)

        # Corrupt
        x_tilde, _ = Pi(x)
        x_tilde = x_tilde.clamp(0.0, 1.0)

        # Recon from corrupted input
        x_hat = model.forward(x_tilde)
        logp = PR.ML_loss(x, x_hat)

        # Poisson fields
        v, grad_v = PR.Estimate_field_grads(x, x_tilde, landmarks=landmarks)
        flux = PR.BC_loss(x, x_tilde, grad_v)

        # Optional bulk term (works here because grad_v lives in pixel space)
        bulk = PR.D_loss(x, x_hat, grad_v)

        loss = logp + lam * (flux + bulk)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 200 == 0:
            print(
                f"step={step} recon={logp.item():.6f} flux={flux.item():.6f} bulk={bulk.item():.6f} loss={loss.item():.6f}"
            )

        if step % viz_every == 0:
            with torch.no_grad():
                fz = model.Encoder(x)  # f(x) on clean
                # source term on corrupted batch (cheaper than recomputing on entire dataset)
                g = jacobian_fro_norm(model.Encoder, x_tilde, create_graph=False)
            visualize_mnist_flat(
                x_clean=x.detach(),
                x_corrupt=x_tilde.detach(),
                x_hat=x_hat.detach(),
                f=fz.detach(),
                g=g.detach(),
                v=v.detach(),
                gradv=grad_v.detach(),
                step=step,
                cfg=viz_cfg,
                title="mnist_flat",
            )

        step += 1


if __name__ == "__main__":
    main()
