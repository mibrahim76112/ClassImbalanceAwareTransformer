import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeCenters(nn.Module):
    def __init__(self, num_classes, feat_dim, momentum=0.95, device='cuda'):
        super().__init__()
        self.m = momentum
        self.register_buffer('centers', torch.zeros(num_classes, feat_dim, device=device))

    @torch.no_grad()
    def update(self, feats, labels):
        for c in labels.unique():
            mask = (labels == c)
            if mask.any():
                mean = feats[mask].mean(dim=0)
                self.centers[c] = self.m * self.centers[c] + (1 - self.m) * mean

    def center_loss(self, feats, labels, per_class_weight=None):
        cy = self.centers[labels]
        d2 = (feats - cy).pow(2).sum(dim=1)
        if per_class_weight is not None:
            w = per_class_weight[labels].to(feats.device)
            return (d2 * w).mean()
        return d2.mean()

class CenterSeparationLoss(nn.Module):
    """Push most-similar rival centers apart by a cosine margin."""
    def __init__(self, K=3, margin=0.28):
        super().__init__()
        self.K = K
        self.margin = margin

    def forward(self, centers: torch.Tensor):
        C = centers.size(0)
        if C <= 1: return centers.new_tensor(0.0)
        Z = F.normalize(centers, dim=1)
        S = Z @ Z.t()
        S = S.clone(); S.fill_diagonal_(-1.0)
        K = min(self.K, max(1, C - 1))
        topk, _ = torch.topk(S, k=K, dim=1)
        thresh = 1.0 - self.margin
        viol = F.relu(topk - thresh)
        return viol.mean()
