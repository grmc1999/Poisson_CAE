"""MNIST (conv encoder/decoder) experiment.

Runs the Poisson-regularized AE where the Poisson potential is built in *latent space*.
Corruption is Gaussian noise injected in pixel space.

Visualizations saved to ./outputs via Utils.visualization.visualize_mnist_conv

Note: In this conv setup we use only the boundary/flux term in latent space
      (no 'bulk' score coupling), because gradv lives in latent coordinates.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from Utils.projectors import CorruptionOperator, CorruptionConfig
from Utils.geometry_estimators import PoissonMCConfig, PoissonMCEstimator
from Utils.visualization import visualize_mnist_conv, VizConfig

from models import AE_model, Poisson_reg_latent, ConvEncoderMNIST, ConvDecoderMNIST


def main(
    *,
    batch_size: int = 128,
    steps: int = 3000,
    lr: float = 2e-4,
    lam: float = 1e-2,
    landmarks: int = 64,
    sigma: float = 0.3,
    z_dim: int = 64,
    hutchinson_samples: int = 1,
    viz_every: int = 500,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Data (keep [0,1])
    tfm = transforms.ToTensor()
    ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)

    # Model
    enc = ConvEncoderMNIST(z_dim=z_dim)
    dec = ConvDecoderMNIST(z_dim=z_dim)
    model = AE_model(enc, dec).to(device)

    # Corruption Πψ: Gaussian noise in pixel space
    Pi = CorruptionOperator(CorruptionConfig(mode="gaussian", sigma=sigma)).to(device)

    # Poisson estimator in latent space (dim=z_dim)
    poisson_est = PoissonMCEstimator(PoissonMCConfig(eps=1e-2, landmarks=landmarks)).to(device)
    PR = Poisson_reg_latent(poisson_est, model, z_dim=z_dim, hutchinson_samples=hutchinson_samples).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    viz_cfg = VizConfig(out_dir="outputs", max_items=8)

    step = 0
    for x, _ in dl:
        if step >= steps:
            break

        x = x.to(device).requires_grad_(True)
        x_tilde, _ = Pi(x)
        x_tilde = x_tilde.clamp(0.0, 1.0)

        # Recon from corrupted input
        x_hat = model.forward(x_tilde)
        logp = PR.ML_loss(x, x_hat)

        # Poisson in latent space
        v, gradv, z_clean, z_tilde, g_land = PR.Estimate_field_grads(x, x_tilde, landmarks=landmarks)
        flux = PR.BC_loss_latent(z_clean, z_tilde, gradv)

        # In this conv setup: only latent flux term
        loss = logp + lam * flux

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 200 == 0:
            print(f"step={step} recon={logp.item():.6f} flux={flux.item():.6f} loss={loss.item():.6f}")

        if step % viz_every == 0:
            # For visualization, show g on the whole batch (not only landmarks)
            from Utils.grad_operations import hutchinson_jacobian_fro_norm
            with torch.no_grad():
                g_batch = hutchinson_jacobian_fro_norm(
                    model.Encoder, x_tilde.detach(), out_dim=z_dim, n_samples=1, create_graph=False
                )

            visualize_mnist_conv(
                x_clean=x.detach(),
                x_corrupt=x_tilde.detach(),
                x_hat=x_hat.detach(),
                z=z_clean.detach(),
                z_tilde=z_tilde.detach(),
                g=g_batch.detach(),
                v=v.detach(),
                gradv=gradv.detach(),
                step=step,
                cfg=viz_cfg,
                title="mnist_conv",
            )

        step += 1


if __name__ == "__main__":
    main()
