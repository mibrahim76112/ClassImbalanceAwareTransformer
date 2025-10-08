# src/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .datasets import TSJitter, TSScale, TSTimeMask

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
    diffusion_sampler=None, synth_ratio=0.0, margin_gate_delta=0.05,
    syn_log: bool = False,   # <-- NEW: logging toggle (default off)
):
    """
    - SupCon/center losses: REAL features only
    - CE loss: REAL (+ optional SYNTHETIC embeddings from diffusion_sampler)
    - Synthetic embeddings are generated in the decision space (z), so we feed them
      directly to the cosine head (no encoder pass).
    """
    model.train()
    ce_loss_fn = LogitAdjustedCrossEntropy(class_counts, tau=tau_la).to(device)
    supcon = WeightedSupConLoss(temperature, class_counts=class_counts).to(device)

    # ArcFace margin warmup
    if hasattr(model, "cos_head"):
        m_max = getattr(model.cos_head, "m", 0.15)
        t = min(1.0, epoch / max(1, total_epochs // 2))
        model.cos_head.m = float(m_max) * (t ** 2)

    lambda_supcon = base_lambda * (epoch / 70.0)
    num_classes = model.cos_head.W.size(0)

    # ---- DEBUG counters for synthetics ----
    syn_proposed_tot = 0        # how many we asked sampler for (sum over steps)
    syn_kept_tot = 0            # how many sampler actually returned (and we used)
    syn_hist = torch.zeros(num_classes, dtype=torch.long)  # kept per-class

    ce_sum = con_sum = n_obs = 0

    for batch in loader:
        # Unpack and move to device
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x = batch[0]
            y = batch[1]
        else:
            x, y = batch
        x = x.to(device, non_blocking=True)        # (B, T, F)
        y = y.to(device, non_blocking=True)
        B = y.size(0)

        # --- On-GPU weak/strong views (for SupCon) ---
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            xw = batch[2].to(device, non_blocking=True)
            xs = batch[3].to(device, non_blocking=True)
        else:
            weak_tfms   = [TSJitter(0.0005), TSScale(0.99, 1.01)]
            strong_tfms = [TSJitter(0.001),  TSScale(0.98, 1.02), TSTimeMask(0.10)]
            xw = x
            for tfn in weak_tfms:
                xw = tfn(xw)
            xs = x
            for tfn in strong_tfms:
                xs = tfn(xs)

        # --- Forward features for REAL batch ---
        feats_w = model.forward_features(xw)        # used for CE / centers
        feats_s = model.forward_features(xs)        # second view for SupCon

        # Project for SupCon; only REAL features participate
        z1 = model.project(feats_w)
        z2 = model.project(feats_s)
        loss_con = supcon(torch.cat([z1, z2], dim=0), y)

        # --- Decide mixup on REAL path ---
        contains_hard = ((y == 15) | (y == 0)).any()
        mixup_prob_eff  = 0.80 if contains_hard else mixup_prob
        mixup_alpha_eff = 0.60 if contains_hard else mixup_alpha
        use_mix = (torch.rand(1).item() < mixup_prob_eff)

        # --- OPTIONAL: sample synthetic embeddings (decision-space) ---
        Z_synth = None
        y_synth = None
        if (diffusion_sampler is not None) and (synth_ratio is not None) and (synth_ratio > 0):
            Ns = int(round(float(synth_ratio) * B))
            if Ns > 0:
                # sample target labels from this batch's labels to match its class mix
                idx = torch.randint(0, B, (Ns,), device=device)
                y_synth = y[idx]
                syn_proposed_tot += Ns

                # IMPORTANT: pass a RAW sampler here (see main train script)
                with torch.no_grad():
                    Z = diffusion_sampler(y=y_synth, n=Ns, steps=None, margin_gate=None)

                if (Z is not None) and (Z.numel() > 0):
                    Z_synth = F.normalize(Z.detach(), dim=-1)
                    kept = int(Z_synth.size(0))
                    y_synth = y_synth[:kept]

                    # per-class kept stats
                    vals, counts = torch.unique(y_synth.detach().cpu(), return_counts=True)
                    for v, c in zip(vals.tolist(), counts.tolist()):
                        syn_hist[v] += int(c)

                    syn_kept_tot += kept
                    if syn_log:
                        print(f"[SYN][train] kept {kept}/{Ns} this step; classes="
                              f"{ {int(v): int(c) for v,c in zip(vals.tolist(), counts.tolist())} }")
                else:
                    if syn_log:
                        print(f"[SYN][train] kept 0/{Ns} this step")

        # --- CE loss (REAL + optional SYNTH) ---
        if use_mix:
            # Mixup on REAL features (soft labels)
            feats_mix, y_mix = latent_mixup(feats_w, y, num_classes=num_classes, alpha=mixup_alpha_eff)
            logits_real = model.cos_head(feats_mix, y=None, use_margin=False)
            loss_ce_real = -(y_mix * F.log_softmax(logits_real, dim=1)).sum(dim=1).mean()

            # Plus CE on synthetic (hard labels) if available
            if Z_synth is not None:
                logits_syn = model.cos_head(Z_synth, y=y_synth, use_margin=True)
                loss_ce_syn = ce_loss_fn(logits_syn, y_synth)
                loss_ce = loss_ce_real + loss_ce_syn
            else:
                loss_ce = loss_ce_real
        else:
     
            logits_real = model.cos_head(feats_w, y=y, use_margin=True)
            if Z_synth is not None:
            
                logits_syn = model.cos_head(Z_synth, y=y_synth, use_margin=True)
                logits_all = torch.cat([logits_real, logits_syn], dim=0)
                y_all      = torch.cat([y,           y_synth   ], dim=0)
                loss_ce = ce_loss_fn(logits_all, y_all)
            else:
                loss_ce = ce_loss_fn(logits_real, y)

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

        # Book-keeping
        ce_sum  += float(loss_ce.detach())  * B
        con_sum += float(loss_con.detach()) * B
        n_obs   += B

    # ---- epoch summary for synthetics ----
    if syn_log and syn_proposed_tot > 0:
        try:
            _hist_list = syn_hist.tolist()
        except Exception:
            _hist_list = [int(x) for x in syn_hist.cpu().numpy().tolist()]
        print(f"[SYN][epoch] proposed={syn_proposed_tot}, kept={syn_kept_tot}; kept-per-class={_hist_list}")

    return ce_sum / max(1, n_obs), con_sum / max(1, n_obs), lambda_supcon
