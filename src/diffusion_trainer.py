import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .diffusion import DecisionSpaceDiffusion, l2_normalize
from torch import amp
from collections import defaultdict  

@torch.no_grad()
def extract_feature_dataset(model, loader, device):
    """
    Old path (kept for small datasets): encodes ALL features into Z (CPU tensor).
   
    """
    Z, Y = [], []
    model.eval()
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x, y = batch[0], batch[1]
        else:
            x, y = batch
        x = x.to(device, non_blocking=True)
     
        f = model.forward_features(x)
        if hasattr(model, "project"):
            f = model.project(f)
        z = F.normalize(f, dim=-1)
        Z.append(z.cpu()); Y.append(y.cpu())
    Z = torch.cat(Z, dim=0)
    Y = torch.cat(Y, dim=0)
    return Z, Y

def train_decision_diffusion(Z, Y, num_classes, *, feat_dim, epochs=5, bs=1024, lr=1e-3, device='cuda',
                             T=1000, steps_infer=20, width=512, depth=3):
    """
    Old trainer that expects a full Z,Y in memory. Fine for small N.
    """
    from torch.utils.data import TensorDataset
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


def train_decision_diffusion_streaming(
    model, feat_loader, device,
    num_classes, feat_dim=None,
    epochs=5, bs=1024, lr=1e-3,
    T=1000, steps_infer=20, width=512, depth=3,
    use_project=True, amp_enabled=True,
    microbatch=256, log_every=200,
    gen_counter=None,   
):
    """
    Streams features from model, trains a decision-space diffusion model,
    and exposes a counting wrapper around ddpm sampling for per-class synthetic usage.
    """
    model.eval()
    ddm = None
    opt = None
    scaler = amp.GradScaler('cuda', enabled=amp_enabled)

   
    if gen_counter is None:
        gen_counter = defaultdict(int)

    step = 0
    for ep in range(1, epochs + 1):
        tot, n = 0.0, 0
        for xb, yb in feat_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.no_grad():
                fb = model.forward_features(xb)
                if use_project and hasattr(model, "project"):
                    fb = model.project(fb)
                zb = F.normalize(fb, dim=-1)  # (B, D)

        
            if ddm is None:
                D = zb.shape[1] if feat_dim is None else int(feat_dim)
                ddm = DecisionSpaceDiffusion(D, num_classes, T=T, num_steps_infer=steps_infer,
                                             width=width, depth=depth).to(device)
                opt = torch.optim.AdamW(ddm.parameters(), lr=lr, weight_decay=1e-4)

        
            B = zb.size(0)
            mb = min(microbatch, B)
            for i in range(0, B, mb):
                zc = zb[i:i+mb]
                yc = yb[i:i+mb]
                with amp.autocast('cuda', enabled=amp_enabled):
                    loss = ddm.ddpm_loss(zc, yc)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                tot += float(loss.detach()) * zc.size(0)
                n += zc.size(0)
                step += 1
                if (step % log_every) == 0:
                    print(f"[Diffusion][ep {ep}] steps={step} avg_loss={tot/max(1,n):.4f}")

        print(f"[Diffusion] epoch {ep}: loss={tot/max(1,n):.4f}")

    ddm.eval()
    _orig_ddim_sample = ddm.ddim_sample  

    @torch.no_grad()
    def ddim_sample(y, n=None, steps=None, margin_gate=None):
        """
        Wrap original sampler to count **synthetic** class-conditional draws.
        We count both the attempted draws (y) and attribute kept samples after margin gating.
        """
        y = y.to(device)
        n_eff = int(n) if n is not None else y.size(0)

        # Count attempted draws per class
        uniq, cnts = torch.unique(y, return_counts=True)
        for cls, cnt in zip(uniq.tolist(), cnts.tolist()):
            gen_counter[int(cls)] += int(cnt)

       
        Zs = _orig_ddim_sample(y=y, n=n_eff, steps=steps or steps_infer, margin_gate=margin_gate)

     
        kept = Zs.size(0)
        if kept < y.size(0):
            tot = int(cnts.sum().item())
            if tot > 0:
                for cls, cnt in zip(uniq.tolist(), cnts.tolist()):
                    share = float(cnt) / float(tot)
                    adj = int(round(share * kept))
                    gen_counter[int(cls)] += adj  

        return Zs

    ddm.ddim_sample = ddim_sample
    ddm._gen_counter = gen_counter  
    return ddm
