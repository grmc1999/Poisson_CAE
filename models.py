import torch
import torch.nn as nn
from Utils.grad_operations import jacobian_fro_norm


class AE_base_model(nn.Module):
    def __init__(self,Encoder,Decoder):
        super().__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def logp(self,x,y):
        raise NotImplementedError

    def score_value(self,x):
        y = self.logp(x)
        logp_grad = torch.autograd.grad(
            outputs=y,
            inputs=x,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return logp_grad

    def forward(self,x):
        x_tilde = self.Decoder(self.Encoder(x))
        return x_tilde

    

class Poisson_reg(nn.Module):
    def __init__(self,PoissonEstimator,model):
        super().__init__()
        self.PoissonEstimator = PoissonEstimator
        #self.Encoder = Encoder
        #self.Decoder = Decoder
        self.model = model
    
    def ML_loss(self,x_clean,x_tilde):
        return self.model.logp(x_clean,x_tilde)
        
    def Estimate_field_grads(self,
        #encoder: nn.Module,
        x_clean: torch.Tensor,
        x_tilde: torch.Tensor,
        landmarks: int = 256,
    ):
        B, d = x_clean.shape

        # Normal induced by corruption map Πψ:
        # nψ(x) = (Πψ(x) - x) / ||Πψ(x) - x||
        #delta = x_tilde - x_clean
        #n = delta / (delta.norm(dim=1, keepdim=True) + 1e-8)  # (B, d)

        # Choose landmark points from the corrupted batch (Monte Carlo quadrature set)
        M = min(landmarks, B)
        idx = torch.randperm(B, device=x_clean.device)[:M]
        x_land = x_tilde[idx].clone().requires_grad_(True)  # (M, d)

        # Source term g(y)=||∇f(y)||_F at landmarks; requires higher-order grads for training
        g_land = jacobian_fro_norm(self.model.Encoder, x_land, create_graph=True)   # (M,)

        # Estimate ∇v at query points x_tilde (the corrupted/OOD points)
        x_query = x_tilde  # (B, d)
        v, gradv = self.PoissonEstimator.forward(x_query, x_land, g_land)  # gradv: (B, d)
        return v, gradv

    def D_loss(self,
                      x_clean: torch.Tensor,
                      x_hat: torch.Tensor,
                      gradv: torch.Tensor,
                      ):
        #z = self.Encoder(x_clean)
        score_value = self.model.score_value(x_clean,x_hat)
        return torch.tensordot(score_value,gradv,dims=([-1],[-1])).mean()


    def BC_loss(self,
        #encoder: nn.Module,
        x_clean: torch.Tensor,
        x_tilde: torch.Tensor,
        #v: torch.Tensor,
        gradv: torch.Tensor,
        #landmarks: int = 256,
    ) -> torch.Tensor:
        """
        x_clean: (B, d)
        x_tilde: (B, d) = Πψ(x_clean)
        returns scalar reg
        """

        B, d = x_clean.shape

        # Normal induced by corruption map Πψ:
        # nψ(x) = (Πψ(x) - x) / ||Πψ(x) - x||
        delta = x_tilde - x_clean
        n = delta / (delta.norm(dim=1, keepdim=True) + 1e-8)  # (B, d)
        reg = x_tilde*(gradv * n).sum(dim=1).mean()
        return reg


class Poisson_reg_latent(nn.Module):
    """Poisson regularizer computed in latent space for image models.

    - Uses Hutchinson to estimate g(y)=||∇_x f(y)||_F at landmark images y.
    - Builds Poisson potential in latent space z=f(x):
        v(z_q) ≈ (1/M) Σ_j G(z_q, z_j) g(y_j)
        ∇v(z_q) similarly.
    - Flux term uses the SAME corruption map through the induced latent displacement:
        n := (z_tilde - z_clean)/||z_tilde - z_clean||
        flux := E[ ∇v(z_tilde) · n ]

    This keeps the experiments stable for MNIST/CIFAR without computing kernels in pixel space.
    """

    def __init__(self, PoissonEstimator, model, z_dim: int, hutchinson_samples: int = 1):
        super().__init__()
        self.PoissonEstimator = PoissonEstimator
        self.model = model
        self.z_dim = int(z_dim)
        self.hutchinson_samples = int(hutchinson_samples)

    def ML_loss(self, x_clean, x_hat):
        return self.model.logp(x_clean, x_hat)

    def Estimate_field_grads(self, x_clean: torch.Tensor, x_tilde: torch.Tensor, landmarks: int = 128):
        from Utils.grad_operations import hutchinson_jacobian_fro_norm

        B = x_clean.size(0)
        M = min(int(landmarks), B)
        idx = torch.randperm(B, device=x_clean.device)[:M]

        # Landmark images (corrupted)
        x_land = x_tilde[idx].clone().requires_grad_(True)
        g_land = hutchinson_jacobian_fro_norm(
            self.model.Encoder,
            x_land,
            out_dim=self.z_dim,
            n_samples=self.hutchinson_samples,
            create_graph=True,
        )  # (M,)

        # Latent points
        z_land = self.model.Encoder(x_land)          # (M, z_dim)
        z_clean = self.model.Encoder(x_clean)        # (B, z_dim)
        z_tilde = self.model.Encoder(x_tilde)        # (B, z_dim)

        # Poisson in latent space
        v, gradv = self.PoissonEstimator.forward(z_tilde, z_land, g_land)  # (B,), (B, z_dim)
        return v, gradv, z_clean, z_tilde, g_land

    def BC_loss_latent(self, x_hat: torch.Tensor, z_clean: torch.Tensor, z_tilde: torch.Tensor, gradv: torch.Tensor) -> torch.Tensor:
        delta = z_tilde - z_clean
        n = delta / (delta.norm(dim=1, keepdim=True) + 1e-8)
        return (x_hat*(gradv * n).sum(dim=1)).mean()
    


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
    

class AE_model(AE_base_model):
    def __init__(self,Encoder,Decoder):
        super().__init__(Encoder,Decoder)

    def forward(self,x):
        x_tilde = self.Decoder(self.Encoder(x))
        return x_tilde
    
    def logp(self,x,x_hat):
        return ((x - x_hat)**2).mean()

    def score_value(self,x,x_hat):
        y = self.logp(x,x_hat)
        logp_grad = torch.autograd.grad(
            outputs=y,
            inputs=x,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return logp_grad


# -----------------------------
# MNIST helpers (flattened + conv)
# -----------------------------

class FlatEncoderMNIST(nn.Module):
    def __init__(self, d_in: int = 784, d_hidden: int = 512, z_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, z_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlatDecoderMNIST(nn.Module):
    def __init__(self, z_dim: int = 64, d_hidden: int = 512, d_out: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_out), nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ConvEncoderMNIST(nn.Module):
    """Simple conv encoder for (B,1,28,28), returns (B,z_dim)."""
    def __init__(self, z_dim: int = 64, base: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 4, 2, 1), nn.SiLU(),      # 28->14
            nn.Conv2d(base, base*2, 4, 2, 1), nn.SiLU(), # 14->7
            nn.Conv2d(base*2, base*4, 3, 2, 1), nn.SiLU(), # 7->4
        )
        self.fc = nn.Linear(base*4 * 4 * 4, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return self.fc(h.flatten(1))


class ConvDecoderMNIST(nn.Module):
    """Simple conv decoder mapping (B,z_dim) -> (B,1,28,28)."""
    def __init__(self, z_dim: int = 64, base: int = 32):
        super().__init__()
        self.fc = nn.Linear(z_dim, base*4 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base*4, base*2, 4, 2, 1), nn.SiLU(),  # 4->8
            nn.ConvTranspose2d(base*2, base, 4, 2, 1), nn.SiLU(),    # 8->16
            nn.ConvTranspose2d(base, 1, 4, 2, 1),                    # 16->32
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        x = self.net(h)
        # center crop 32->28
        return x[:, :, 2:30, 2:30]

