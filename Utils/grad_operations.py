import torch
import torch.nn as nn
import math

def jacobian_fro_norm(f: nn.Module, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """
    x: (B, d), requires_grad=True
    f(x): (B, m)
    returns g: (B,) where g_i = ||J_f(x_i)||_F
    """
    if not x.requires_grad:
        x = x.requires_grad_(True)
    y = f(x)  # (B, m)
    B, m = y.shape
    J_sq = torch.zeros(B, device=x.device, dtype=x.dtype)
    for k in range(m):
        grad_k = torch.autograd.grad(
            outputs=y[:, k].sum(),
            inputs=x,
            create_graph=create_graph,
            retain_graph=True,
            only_inputs=True,
        )[0]  # (B, d)
        J_sq = J_sq + (grad_k ** 2).sum(dim=1)
    return torch.sqrt(J_sq + 1e-12)


def hutchinson_jacobian_fro_norm(
    f: nn.Module,
    x: torch.Tensor,
    out_dim: int,
    n_samples: int = 1,
    create_graph: bool = True,
) -> torch.Tensor:
    """Estimate g(x) = ||J_f(x)||_F using Hutchinson.

    This avoids forming the full Jacobian when either input dimension (e.g. images)
    or output dimension is large.

    For f: R^d -> R^m, we have:
      ||J||_F^2 = E_u ||J^T u||^2  with u ~ N(0, I_m)
    and J^T u = ∇_x <u, f(x)>.

    Args:
      f: module
      x: input tensor, shape (B, ...) with requires_grad=True recommended
      out_dim: m (dimension of f(x) last dim)
      n_samples: number of Hutchinson probe vectors
      create_graph: keep graph for higher-order grads (needed if training with this regularizer)

    Returns:
      g: (B,) tensor
    """
    if not x.requires_grad:
        x = x.requires_grad_(True)

    y = f(x)
    if y.dim() != 2 or y.size(1) != out_dim:
        raise ValueError(f"Expected f(x) to have shape (B,{out_dim}), got {tuple(y.shape)}")

    B = y.size(0)
    acc = torch.zeros(B, device=x.device, dtype=x.dtype)
    for _ in range(int(n_samples)):
        u = torch.randn(B, out_dim, device=x.device, dtype=x.dtype)
        s = (y * u).sum(dim=1).sum()  # scalar
        grad_x = torch.autograd.grad(
            outputs=s,
            inputs=x,
            create_graph=create_graph,
            retain_graph=True,
            only_inputs=True,
        )[0]
        acc = acc + grad_x.flatten(1).pow(2).sum(dim=1)

    fro_sq = acc / float(n_samples)
    return torch.sqrt(fro_sq + 1e-12)


def omega_d(d: int) -> float:
    return 2.0 * (math.pi ** (d / 2.0)) / math.gamma(d / 2.0)

def green_reg(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """
    x: (B, d), y: (M, d) => G: (B, M)
    """
    B, d = x.shape
    r2 = ((x[:, None, :] - y[None, :, :]) ** 2).sum(dim=2)
    r = torch.sqrt(r2 + eps**2)
    if d == 2:
        return -(1.0 / (2.0 * math.pi)) * torch.log(r)
    if d >= 3:
        c = 1.0 / ((d - 2.0) * omega_d(d))
        return c * (r ** (2.0 - d))
    raise ValueError("d must be >=2")

def gradx_green_reg(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """
    ∇_x G(x,y): returns (B, M, d)
    """
    B, d = x.shape
    diff = x[:, None, :] - y[None, :, :]
    r2 = (diff ** 2).sum(dim=2)
    r = torch.sqrt(r2 + eps**2)
    if d == 2:
        # G = -(1/(2π)) log r  => ∇ = -(1/(2π)) diff / (r^2)
        return -(1.0 / (2.0 * math.pi)) * diff / (r2[..., None] + eps**2)
    if d >= 3:
        c = 1.0 / ((d - 2.0) * omega_d(d))
        return c * (2.0 - d) * diff * (r[..., None] ** (-d))
    raise ValueError("d must be >=2") 
