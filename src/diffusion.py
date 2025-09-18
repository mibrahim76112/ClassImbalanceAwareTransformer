# src/diffusion.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def l2_normalize(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def timestep_embedding(timesteps, dim):
    # sinusoidal (stable & simple)
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:  # pad
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class CondEpsPredictor(nn.Module):
    def __init__(self, feat_dim, num_classes, width=512, depth=3, tdim=128, ydim=64, extra_cond_dim=0):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.t_proj = nn.Linear(tdim, width)
        self.y_emb = nn.Embedding(num_classes, ydim)
        self.extra_proj = nn.Linear(extra_cond_dim, ydim) if extra_cond_dim > 0 else None

        in_dim = feat_dim + width + ydim + (ydim if self.extra_proj else 0)
        layers = []
        for i in range(depth):
            layers += [nn.Linear(in_dim if i == 0 else width, width), nn.GELU()]
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(width, feat_dim)

        
        self.margin_head = nn.Sequential(nn.Linear(width, 1), nn.Tanh())

    def forward(self, z_noisy, t, y, extra=None):
        te = self.t_proj(timestep_embedding(t, self.t_proj.in_features))
        ye = self.y_emb(y)
        if self.extra_proj is not None and extra is not None:
            ee = self.extra_proj(extra)
            h = torch.cat([z_noisy, te, ye, ee], dim=-1)
        else:
            h = torch.cat([z_noisy, te, ye], dim=-1)
        h = self.net(h)
        eps = self.out(h)
        mg_score = self.margin_head(h)  
        return eps, mg_score

def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)

# --------- DDPM loss + DDIM sampler on decision sphere
class DecisionSpaceDiffusion(nn.Module):
    def __init__(self, feat_dim, num_classes, T=1000, num_steps_infer=20, width=512, depth=3):
        super().__init__()
        self.feat_dim = feat_dim
        self.T = int(T)
        self.num_steps_infer = int(num_steps_infer)
        self.model = CondEpsPredictor(feat_dim, num_classes, width=width, depth=depth)
        self.register_buffer("betas", cosine_beta_schedule(self.T), persistent=False)
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod), persistent=False)

    def ddpm_loss(self, z0, y, extra=None):
        """
        z0: (B, D) unit-norm embeddings from frozen encoder
        y : (B,)
        """
        B = z0.size(0)
        t = torch.randint(0, self.T, (B,), device=z0.device, dtype=torch.long)
        noise = torch.randn_like(z0)
        zt = self.sqrt_alphas_cumprod[t].unsqueeze(1) * z0 + self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1) * noise
        pred_eps, _ = self.model(zt, t, y, extra=extra)
        return F.mse_loss(pred_eps, noise)

    @torch.no_grad()
    def ddim_sample(self, y, n, steps=None, extra=None, margin_gate=None):
        """
        y: (n,) target classes. Returns unit-norm embeddings (n, D).
        margin_gate: callable(z_hat, y)->Bool mask to accept/reject samples
        """
        steps = steps or self.num_steps_infer
        device = next(self.model.parameters()).device
        z = torch.randn(n, self.feat_dim, device=device)
        t_seq = torch.linspace(self.T - 1, 0, steps, device=device).long()
        for ti in t_seq:
            eps, _ = self.model(z, ti.expand(n), y, extra=extra)
            a_bar = self.alphas_cumprod[ti]
            z0 = (z - torch.sqrt(1 - a_bar) * eps) / torch.sqrt(a_bar + 1e-8)
            z = z0  
            z = l2_normalize(z) 
        
        if margin_gate is not None:
            keep = margin_gate(z, y)  
            z = z[keep]
        return z
