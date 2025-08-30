import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

class TSJitter:
    def __init__(self, sigma=0.01): self.sigma = sigma
    def __call__(self, x): return x + torch.randn_like(x) * self.sigma

class TSScale:
    def __init__(self, low=0.9, high=1.1): self.low, self.high = low, high
    def __call__(self, x):
        s = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(self.low, self.high)
        return x * s

class TSTimeMask:
    def __init__(self, max_frac=0.10): self.max_frac = max_frac
    def __call__(self, x):
        B, T, F = x.shape
        L = max(1, int(T * self.max_frac))
        t0 = torch.randint(0, max(1, T - L + 1), (1,), device=x.device).item()
        x = x.clone(); x[:, t0:t0+L, :] = 0
        return x

class TwoCropsTransform:
    def __init__(self, weak_tfms, strong_tfms):
        self.weak, self.strong = weak_tfms, strong_tfms
    def __call__(self, x):
        xw, xs = x, x
        for t in self.weak:   xw = t(xw)
        for t in self.strong: xs = t(xs)
        return xw, xs

class ContrastiveTSDataset(Dataset):
    def __init__(self, X, y, two_crops):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.two_crops = two_crops
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        x = self.X[idx].unsqueeze(0)
        y = self.y[idx]
        xw, xs = self.two_crops(x)
        return x.squeeze(0), y, xw.squeeze(0), xs.squeeze(0)

class PlainTSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def make_sampler(y):
    y = y.astype(int)
    class_counts = np.bincount(y)
    class_counts[class_counts == 0] = 1
    weights = 1.0 / class_counts[y]
    sampler = WeightedRandomSampler(weights, num_samples=len(y), replacement=True)
    return sampler, class_counts
