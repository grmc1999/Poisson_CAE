import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.grad_operations import jacobian_fro_norm


class AE_base_model(nn.Module):
    def __init__(self,Encoder,Decoder):
        super().__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def logp(self, y_true, y_pred):
        """Downstream loss (negative log-likelihood / risk).

        For reconstruction tasks, y_true is the clean input x and y_pred is x_hat.
        For supervised tasks, y_true is the target (labels or regression targets)
        and y_pred is the model output.
        """
        raise NotImplementedError

    def score_value(self, x, y_true, y_pred):
        """Gradient of downstream loss w.r.t. the *input* x.

        This is the object used in the bulk term: <∇_x log p_D(x), ∇v(x)>.
        In our implementation, we treat the downstream risk as a surrogate.
        """
        y = self.logp(y_true, y_pred)
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
    
    def ML_loss(self, y_true, y_pred):
        return self.model.logp(y_true, y_pred)
        
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
                      y_true: torch.Tensor,
                      y_pred: torch.Tensor,
                      gradv: torch.Tensor,
                      ):
        #z = self.Encoder(x_clean)
        score_value = self.model.score_value(x_clean, y_true, y_pred)
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
        reg = (gradv * n).sum(dim=1).mean()
        return reg
    


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
    
    def logp(self, x_true, x_hat):
        return ((x_true - x_hat)**2).mean()


class Classifier_model(AE_base_model):
    """Encoder + linear head classifier.

    We keep the name conventions of AE_base_model for compatibility with Poisson_reg.
    Decoder is replaced by a classification head.
    """
    def __init__(self, Encoder: nn.Module, n_classes: int):
        super().__init__(Encoder, nn.Identity())
        # infer z dim by a lazy linear
        self.head = nn.LazyLinear(n_classes)

    def forward(self, x):
        z = self.Encoder(x)
        return self.head(z)  # logits

    def logp(self, y_true, y_pred):
        # y_true: (B,) int64 labels, y_pred: (B,C) logits
        return F.cross_entropy(y_pred, y_true)


class Regressor_model(AE_base_model):
    """Encoder + MLP head regression."""
    def __init__(self, Encoder: nn.Module, out_dim: int, hidden: int = 64):
        super().__init__(Encoder, nn.Identity())
        self.head = nn.Sequential(
            nn.LazyLinear(hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        z = self.Encoder(x)
        return self.head(z)

    def logp(self, y_true, y_pred):
        # mean-squared error
        return ((y_true - y_pred) ** 2).mean()


class GRUEncoder(nn.Module):
    """GRU-based encoder for flattened sequences.

    Expects x shaped (B, T) or (B, T, Din). If x is (B, T), we assume Din=1.
    Returns z shaped (B, z_dim).

    This is intended for the time-series experiment while keeping the rest of
    the repo (Poisson estimator / corruption / training loop) unchanged.
    """
    def __init__(self, T: int, din: int = 1, hidden: int = 64, z_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.T = T
        self.din = din
        self.gru = nn.GRU(input_size=din, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            # (B,T) -> (B,T,1)
            x = x.unsqueeze(-1)
        elif x.dim() != 3:
            raise ValueError(f"GRUEncoder expects (B,T) or (B,T,D), got {tuple(x.shape)}")
        # If flattened length differs, try to reshape safely.
        if x.size(1) != self.T:
            raise ValueError(f"Expected sequence length T={self.T}, got {x.size(1)}")
        h, _ = self.gru(x)
        last = h[:, -1, :]
        return self.proj(last)

