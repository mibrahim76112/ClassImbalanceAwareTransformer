# src/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitAdjustedCrossEntropy(nn.Module):
    """
    LA-CE for class imbalance
    Prior logits = tau * log(class_freq)
    """
    def __init__(self, class_counts, tau=1.0):
        super().__init__()
        cc = torch.tensor(class_counts, dtype=torch.float)
        cc[cc == 0] = 1.0
        prior = cc / cc.sum()
        self.register_buffer("prior_logits", tau * torch.log(prior))

    def forward(self, logits, y):
        la_logits = logits + self.prior_logits
        return F.cross_entropy(la_logits, y)


class WeightedSupConLoss(nn.Module):
    """Supervised Contrastive Loss with optional class-frequency weighting."""
    def __init__(self, temperature=0.12, class_counts=None):
        super().__init__()
        self.t = temperature
        if class_counts is not None:
            cc = torch.tensor(class_counts, dtype=torch.float)
            w = 1.0 / cc
            self.register_buffer("class_w", (w / w.sum()) * len(cc))
        else:
            self.class_w = None

    def forward(self, feats2views, labels):
        # feats2views: (2B, d), labels: (B,)
        feats = F.normalize(feats2views, dim=1)
        B = labels.size(0)
        lab2 = labels.repeat_interleave(2)

        sim = feats @ feats.t() / self.t
        mask = torch.ones_like(sim, dtype=torch.bool)
        mask.fill_diagonal_(False)

        same = (lab2.unsqueeze(0) == lab2.unsqueeze(1))
        pos = same & mask

        sim_exp = torch.exp(sim) * mask
        denom = sim_exp.sum(dim=1) + 1e-12
        num = (sim_exp * pos).sum(dim=1) + 1e-12
        loss_i = -torch.log(num / denom)

        valid = (pos.sum(dim=1) > 0).float()
        if self.class_w is not None:
            w = self.class_w[lab2].to(feats.device)
            loss = (loss_i * w * valid).sum() / (w * valid).sum().clamp_min(1.0)
        else:
            loss = (loss_i * valid).sum() / valid.sum().clamp_min(1.0)
        return loss


def latent_mixup(feats, y, num_classes, alpha=0.4):
    if alpha is None or alpha <= 0:
        return feats, F.one_hot(y, num_classes=num_classes).float()
    B = feats.size(0); device = feats.device
    lam = torch.distributions.Beta(alpha, alpha).sample((B,)).to(device)
    lam = torch.maximum(lam, 1.0 - lam)
    idx = torch.randperm(B, device=device)

    feats_mix = lam.view(B, 1) * feats + (1 - lam).view(B, 1) * feats[idx]
    y_one = F.one_hot(y, num_classes=num_classes).float()
    y_perm = F.one_hot(y[idx], num_classes=num_classes).float()
    y_soft = lam.view(B, 1) * y_one + (1 - lam).view(B, 1) * y_perm
    return feats_mix, y_soft


def train_one_epoch(
    model, loader, optimizer, device, class_counts,
    tau_la=1.0, base_lambda=0.5, temperature=0.12,
    epoch=1, total_epochs=20,
    mixup_alpha=0.4, mixup_prob=0.35,
    centers=None, per_class_center_w=None, lambda_center=0.010,
    center_sep=None, lambda_center_sep=0.010,
    diffusion_sampler=None, synth_ratio=0.0, margin_gate_delta=0.05  # NEW
):
    model.train()
    ce_loss_fn = LogitAdjustedCrossEntropy(class_counts, tau=tau_la).to(device)
    supcon = WeightedSupConLoss(temperature, class_counts=class_counts).to(device)

    # warm-up margin (0 -> m)
    if hasattr(model, "cos_head"):
        m_max = getattr(model.cos_head, "m", 0.15)
        t = min(1.0, epoch / max(1, total_epochs // 2))
        model.cos_head.m = float(m_max) * (t ** 2)

    lambda_supcon = base_lambda * (epoch / 70)
    num_classes = model.cos_head.W.size(0)

    import numpy as _np
    counts = _np.bincount(_np.arange(num_classes), minlength=num_classes)  # dummy to keep API stable
    # We'll select minority classes dynamically from class_counts:
    cc_np = _np.array(class_counts if len(class_counts) == num_classes else _np.pad(class_counts, (0, num_classes-len(class_counts))))
    minority = [i for i,c in enumerate(cc_np) if c > 0 and c < cc_np.max()]

    def margin_gate(z_hat, y):
        # accept only if cos_y >= max_other + delta (delta ~ margin target)
        with torch.no_grad():
            logits = model.cos_head(z_hat, y=None, use_margin=False) / max(model.cos_head.s, 1.0)
            tgt = logits[torch.arange(logits.size(0), device=logits.device), y]
            logits[torch.arange(logits.size(0), device=logits.device), y] = -1e9
            rival, _ = logits.max(dim=1)
            ok = (tgt >= rival + margin_gate_delta)
        return ok

    ce_sum = con_sum = n_obs = 0
    for x, y, xw, xs in loader:
        # --------- NEW: synthetic mini-step (before real step), prob = synth_ratio
        if (diffusion_sampler is not None) and (synth_ratio > 0.0) and (torch.rand(1).item() < synth_ratio) and (len(minority) > 0):
            Bsyn = min(128, xw.size(0))  # small synthetic step
            ys = torch.tensor(_np.random.choice(minority, size=Bsyn), device=device).long()
            with torch.no_grad():
                z_syn = diffusion_sampler.ddim_sample(ys, n=Bsyn, steps=None, margin_gate=margin_gate)
                if z_syn.numel() > 0:
                    ys = ys[:z_syn.size(0)]
                    # CE on synthetic embeddings via cosine head
                    logits_syn = model.cos_head(z_syn, y=ys, use_margin=True)
                    loss_syn = ce_loss_fn(logits_syn, ys)
                    optimizer.zero_grad(); loss_syn.backward(); optimizer.step()

        # --------- ORIGINAL REAL-BATCH PATH (unchanged)
        xw, xs, y = xw.to(device), xs.to(device), y.to(device)

        feats_w = model.forward_features(xw)
        feats_s = model.forward_features(xs)

        z1 = model.project(feats_w)
        z2 = model.project(feats_s)
        loss_con = supcon(torch.cat([z1, z2], dim=0), y)

        contains_hard = ((y == 15) | (y == 0)).any()
        mixup_prob_eff  = 0.80 if contains_hard else mixup_prob
        mixup_alpha_eff = 0.60 if contains_hard else mixup_alpha
        use_mix = (torch.rand(1).item() < mixup_prob_eff)

        if use_mix:
            feats_mix, y_mix = latent_mixup(feats_w, y, num_classes=num_classes, alpha=mixup_alpha_eff)
            logits = model.cos_head(feats_mix, y=None, use_margin=False)
            loss_ce = -(y_mix * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        else:
            logits = model.cos_head(feats_w, y=y, use_margin=True)
            loss_ce = ce_loss_fn(logits, y)

        loss_center_term = 0.0
        if centers is not None:
            with torch.no_grad():
                centers.update(feats_w.detach(), y.detach())
            loss_center_term = centers.center_loss(feats_w, y, per_class_weight=per_class_center_w)

        loss_center_sep_term = 0.0
        if (center_sep is not None) and (centers is not None):
            loss_center_sep_term = center_sep(centers.centers)

        loss = (loss_ce
                + lambda_supcon * loss_con
                + lambda_center * loss_center_term
                + lambda_center_sep * loss_center_sep_term)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        b = y.size(0)
        ce_sum  += float(loss_ce.detach())  * b
        con_sum += float(loss_con.detach()) * b
        n_obs   += b

    return ce_sum / max(1, n_obs), con_sum / max(1, n_obs), lambda_supcon
