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
        print(logp_grad.shape)
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
                      gradv: torch.Tensor,
                      ):
        #z = self.Encoder(x_clean)
        score_value = self.model.score_value(x_clean)
        return torch.tensordor(score_value,gradv,dim=1)


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

        ## Choose landmark points from the corrupted batch (Monte Carlo quadrature set)
        #M = min(landmarks, B)
        #idx = torch.randperm(B, device=x_clean.device)[:M]
        #x_land = x_tilde[idx].clone().requires_grad_(True)  # (M, d)
#
        ## Source term g(y)=||∇f(y)||_F at landmarks; requires higher-order grads for training
        #g_land = jacobian_fro_norm(encoder, x_land, create_graph=True)   # (M,)
#
        ## Estimate ∇v at query points x_tilde (the corrupted/OOD points)
        #x_query = x_tilde  # (B, d)
        #_, gradv = self.PoissonEstimator(x_query, x_land, g_land)  # gradv: (B, d)

        #v, gradv = self.Estimate_field_grads(x_clean, x_tilde,landmarks)

        # Boundary-like flux term
        print(gradv.shape)
        print(n.shape)
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
    
    def logp(self,x,x_hat):
        return ((x - x_hat)**2).mean()

    def score_value(self,x):
        y = self.logp(x)
        logp_grad = torch.autograd.grad(
            outputs=y,
            inputs=x,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        print(logp_grad.shape)
        return logp_grad

