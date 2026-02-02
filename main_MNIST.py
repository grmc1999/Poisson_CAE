"""MNIST runner (carefully updated).

This script keeps the repo's training logic (Poisson_reg + PoissonMCEstimator)
but fixes several issues that previously made MNIST runs incorrect/fragile:

Fixes / upgrades:
  - Correctly reconstructs *clean* x from *corrupted* x_tilde.
  - Calls D_loss with the proper signature (x_clean, y_true, y_pred, gradv).
  - Adds image-space perturbations for MNIST: gaussian, rotate, zoom, mask, fgsm.
  - Ensures shapes are consistent: corruption happens on images, Poisson terms use
    flattened vectors.
  - Uses the unified visualize_fields() (PCA-based for high-D inputs).

Usage examples:
  python main_MNIST.py --steps 2000 --corruption gaussian --sigma 0.3
  python main_MNIST.py --corruption rotate --rot_max_deg 30
  python main_MNIST.py --corruption zoom --zoom_min 0.8 --zoom_max 1.2
  python main_MNIST.py --corruption mask --mask_prob 0.2
  python main_MNIST.py --corruption fgsm --fgsm_eps 0.05
"""

import os
import argparse
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from Utils.projectors import CorruptionConfig, CorruptionOperator, fgsm_perturb
from Utils.geometry_estimators import PoissonMCConfig, PoissonMCEstimator
from Utils.visualization import visualize_fields, VizConfig

from models import Poisson_reg, AE_model, Encoder, Decoder


# -----------------------------
# MNIST image-space perturbations
# -----------------------------
class MnistImageProjector(nn.Module):
    """Image-space projector for MNIST.

    Uses CorruptionConfig to support:
      - gaussian: additive Gaussian noise
      - rotate: random rotation
      - zoom: random scaling (centered)
      - mask: coordinate masking (pixel dropout)
      - fgsm: handled in the training loop (needs model/loss)

    Returns x_tilde_img with same shape as x_img: (B,1,28,28).
    """

    def __init__(self, cfg: CorruptionConfig):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        mode = self.cfg.mode
        if mode == "gaussian":
            x = x_img + self.cfg.sigma * torch.randn_like(x_img)
            return x.clamp(0.0, 1.0)

        if mode == "mask":
            # pixel dropout: zero random coordinates
            if self.cfg.mask_prob <= 0:
                return x_img
            keep = (torch.rand_like(x_img) > self.cfg.mask_prob).to(x_img.dtype)
            x = x_img * keep + (1.0 - keep) * float(self.cfg.mask_value)
            return x

        if mode in ("rotate", "zoom"):
            # torchvision functional transforms operate per-sample (loop is OK for MNIST)
            out = []
            for i in range(x_img.size(0)):
                img = x_img[i]
                if mode == "rotate":
                    # sample angle in degrees
                    angle = float((2.0 * torch.rand((), device=x_img.device) - 1.0) * self.cfg.rot_max_deg)
                    img2 = TF.rotate(img, angle=angle, interpolation=TF.InterpolationMode.BILINEAR)
                else:
                    # zoom via affine with scale
                    scale = float(torch.empty((), device=x_img.device).uniform_(self.cfg.zoom_min, self.cfg.zoom_max))
                    img2 = TF.affine(
                        img,
                        angle=0.0,
                        translate=[0, 0],
                        scale=scale,
                        shear=[0.0, 0.0],
                        interpolation=TF.InterpolationMode.BILINEAR,
                    )
                out.append(img2)
            x = torch.stack(out, dim=0).clamp(0.0, 1.0)
            return x

        if mode == "fgsm":
            # handled in training loop (needs model + y_true)
            return x_img

        # fall back: try vector-space corruption (not recommended for images)
        return x_img


def make_mnist_loader(batch_size: int) -> DataLoader:
    tfm = T.Compose([T.ToTensor()])
    ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)


def run_mnist(
    *,
    device: str,
    steps: int,
    batch_size: int,
    lr: float,
    lam: float,
    landmarks: int,
    viz_every: int,
    viz_dir: str,
    corruption_cfg: CorruptionConfig,
    z_dim: int = 64,
    hidden: int = 512,
):
    os.makedirs(viz_dir, exist_ok=True)
    loader = make_mnist_loader(batch_size)

    d = 28 * 28
    model = AE_model(
        Encoder(d=d, h=hidden, z=z_dim),
        Decoder(z=z_dim, h=hidden, d=d),
    ).to(device)

    # We keep the repo's PoissonMCEstimator API.
    # NOTE: For d=784, the free-space Green kernel is not "physically" ideal.
    # This runner is meant as a practical experiment harness.
    poisson_est = PoissonMCEstimator(PoissonMCConfig(eps=1e-2, landmarks=landmarks)).to(device)
    PR = Poisson_reg(poisson_est, model)

    # Projector: image-space for rotate/zoom; vector-space for others.
    img_projector = MnistImageProjector(corruption_cfg).to(device)
    vec_projector = CorruptionOperator(corruption_cfg).to(device)

    class _VizProjector(nn.Module):
        """Projector wrapper for visualization.

        visualize_fields() expects a projector that maps flattened x -> (x_tilde, t).
        For rotate/zoom we must operate in image space, then flatten back.
        """

        def __init__(self, cfg: CorruptionConfig):
            super().__init__()
            self.cfg = cfg

        @torch.no_grad()
        def forward(self, x_flat: torch.Tensor):
            if self.cfg.mode in ("rotate", "zoom", "mask", "gaussian"):
                x_img = x_flat.view(-1, 1, 28, 28)
                x_tilde_img = img_projector(x_img)
                return x_tilde_img.view(x_flat.size(0), -1), None
            if self.cfg.mode == "fgsm":
                # For visualization, use the vector-space helper too.
                x_adv = fgsm_perturb(model, x_flat, y_true=x_flat, eps=self.cfg.fgsm_eps, task="reconstruction")
                return x_adv, None
            x_tilde, t = vec_projector(x_flat)
            return x_tilde, t

    viz_projector = _VizProjector(corruption_cfg).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    step = 0
    for epoch in range(10**9):
        for x_img, _ in loader:
            x_img = x_img.to(device)

            # --- corruption Πψ in *image space* ---
            if corruption_cfg.mode in ("rotate", "zoom", "mask", "gaussian"):
                x_tilde_img = img_projector(x_img)
            elif corruption_cfg.mode == "fgsm":
                # FGSM needs the model and target; do it in vector space using reconstruction loss
                x_vec = x_img.view(x_img.size(0), -1)
                x_adv = fgsm_perturb(model, x_vec, y_true=x_vec, eps=corruption_cfg.fgsm_eps, task="reconstruction")
                x_tilde_img = x_adv.view_as(x_img)
            else:
                # fallback: apply vector corruption on flattened then reshape
                x_vec = x_img.view(x_img.size(0), -1)
                x_tilde_vec, _ = vec_projector(x_vec)
                x_tilde_img = x_tilde_vec.view_as(x_img)

            # flatten for Poisson_reg + AE model
            x = x_img.view(x_img.size(0), -1).requires_grad_(True)                 # clean
            x_tilde = x_tilde_img.view(x_tilde_img.size(0), -1).detach()          # corrupted
            x_tilde = x_tilde.requires_grad_(True)

            # --- downstream task: reconstruct clean x from corrupted x_tilde ---
            x_hat = model(x_tilde)
            logp = PR.ML_loss(x, x_hat)

            # --- Poisson fields ---
            v, grad_v = PR.Estimate_field_grads(x_clean=x, x_tilde=x_tilde, landmarks=landmarks)
            flux = PR.BC_loss(x_clean=x, x_tilde=x_tilde, gradv=grad_v)
            bulk = PR.D_loss(x_clean=x, y_true=x, y_pred=x_hat, gradv=grad_v)

            loss = logp + lam * (flux + bulk)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            step += 1
            if step % 100 == 0:
                print(
                    f"[mnist] step={step} recon={logp.item():.6f} "
                    f"flux={flux.item():.6f} bulk={bulk.item():.6f} loss={loss.item():.6f}"
                )

            if viz_every > 0 and (step % viz_every == 0):
                try:
                    # visualize on clean batch (flattened)
                    xb = x.detach()
                    visualize_fields(
                        model=model,
                        poisson_reg=PR,
                        projector=viz_projector,
                        x_batch=xb,
                        out_dir=viz_dir,
                        step=step,
                        device=device,
                        cfg=VizConfig(landmarks=landmarks, dpi=160),
                    )
                except Exception as e:
                    print(f"[mnist-viz] warning at step {step}: {e}")

            if step >= steps:
                return


def build_corruption_cfg(args: argparse.Namespace) -> CorruptionConfig:
    return CorruptionConfig(
        mode=args.corruption,
        sigma=args.sigma,
        rot_max_deg=args.rot_max_deg,
        zoom_min=args.zoom_min,
        zoom_max=args.zoom_max,
        mask_prob=args.mask_prob,
        mask_value=args.mask_value,
        fgsm_eps=args.fgsm_eps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lam", type=float, default=1e-2)
    parser.add_argument("--landmarks", type=int, default=64)
    parser.add_argument("--viz_every", type=int, default=500)
    parser.add_argument("--viz_dir", type=str, default="outputs_mnist")

    # Corruption
    parser.add_argument(
        "--corruption",
        type=str,
        default="gaussian",
        choices=["gaussian", "rotate", "zoom", "mask", "fgsm"],
    )
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--rot_max_deg", type=float, default=30.0)
    parser.add_argument("--zoom_min", type=float, default=0.85)
    parser.add_argument("--zoom_max", type=float, default=1.15)
    parser.add_argument("--mask_prob", type=float, default=0.2)
    parser.add_argument("--mask_value", type=float, default=0.0)
    parser.add_argument("--fgsm_eps", type=float, default=0.05)

    # Model sizes
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=512)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_mnist(
        device=device,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lam=args.lam,
        landmarks=args.landmarks,
        viz_every=args.viz_every,
        viz_dir=args.viz_dir,
        corruption_cfg=build_corruption_cfg(args),
        z_dim=args.z_dim,
        hidden=args.hidden,
    )
