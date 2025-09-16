# src/diffusion_trainer.py
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from .diffusion import DecisionSpaceDiffusion, l2_normalize

@torch.no_grad()
def extract_feature_dataset(model, loader, device):
    Z, Y = [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x, y = batch[0], batch[1]
        else:
            x, y = batch
        z = F.normalize(model.forward_features(x.to(device)), dim=-1)
        Z.append(z.cpu()); Y.append(y.cpu())
    Z = torch.cat(Z, dim=0)
    Y = torch.cat(Y, dim=0)
    return Z, Y

def train_decision_diffusion(Z, Y, num_classes, *, feat_dim, epochs=5, bs=1024, lr=1e-3, device='cuda',
                             T=1000, steps_infer=20, width=512, depth=3):
    ds = TensorDataset(Z, Y)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
    ddm = DecisionSpaceDiffusion(feat_dim, num_classes, T=T, num_steps_infer=steps_infer,
                                 width=width, depth=depth).to(device)
    opt = torch.optim.AdamW(ddm.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(1, epochs + 1):
        tot, n = 0.0, 0
        for z, y in dl:
            z = l2_normalize(z.to(device))
            y = y.to(device)
            loss = ddm.ddpm_loss(z, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * z.size(0); n += z.size(0)
        print(f"[Diffusion] epoch {ep}: loss={tot/max(1,n):.4f}")
    return ddm
