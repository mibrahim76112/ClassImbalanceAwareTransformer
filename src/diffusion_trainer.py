import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .diffusion import DecisionSpaceDiffusion, l2_normalize
from torch import amp
from collections import defaultdict
from typing import Dict, Optional


@torch.no_grad()
def extract_feature_dataset(model, loader, device):
    """
    Old path (kept for small datasets): encodes ALL features into Z (CPU tensor).
    Avoid for large datasets; prefer train_decision_diffusion_streaming below.
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


def train_decision_diffusion(
    Z, Y, num_classes, *, feat_dim, epochs=5, bs=1024, lr=1e-3, device='cuda',
    T=1000, steps_infer=20, width=512, depth=3
):
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
    gen_counter: Optional[Dict[int, int]] = None,
    quota: Optional[Dict[int, int]] = None,
    auto_balance_to_majority: bool = False,
    max_gen_chunk: int = 8192,
):
    """
    Streams features from model, trains a decision-space diffusion model, and
    exposes a wrapper around sampling that (a) enforces per-class quotas and
    (b) optionally auto-balances each class up to the majority count.

    - gen_counter: defaultdict(int) to accumulate KEPT synthetic counts per class.
    - quota: dict[class] -> remaining budget to reach majority count.
    - auto_balance_to_majority: if True, the first time sampling is called we
      proactively generate all remaining quota (in chunks), independent of synth_ratio.
    - max_gen_chunk: caps the size of any single generation call for memory safety.
    """
    model.eval()
    ddm = None
    opt = None
    scaler = amp.GradScaler('cuda', enabled=amp_enabled)

    if gen_counter is None:
        gen_counter = defaultdict(int)

    # normalize quota to defaultdict(int)
    if quota is None:
        quota_dd = defaultdict(lambda: 10**12)  # effectively infinite if not provided
    else:
        quota_dd = defaultdict(int)
        for k, v in quota.items():
            quota_dd[int(k)] = int(v)
    quota = quota_dd

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

            # Lazy-init diffusion on first real batch
            if ddm is None:
                D = zb.shape[1] if feat_dim is None else int(feat_dim)
                ddm = DecisionSpaceDiffusion(D, num_classes, T=T, num_steps_infer=steps_infer,
                                             width=width, depth=depth).to(device)
                opt = torch.optim.AdamW(ddm.parameters(), lr=lr, weight_decay=1e-4)

            # Micro-batch to prevent OOM
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
    _orig_ddim_sample = ddm.ddim_sample  # save original to avoid recursion
    ddm.ddim_sample_raw = _orig_ddim_sample  # <-- NEW: expose unrestricted sampler for offline export
    _auto_filled_once = False

    def _attribute_and_decrement(cap_map, kept_total: int):
        """
        Proportionally attribute KEPT samples across classes and decrement:
          - gen_counter (kept per class)
          - quota (remaining per class)
        Returns a dict kept_map[class] = kept_kept_for_that_class.
        """
        kept_map = {int(c): 0 for c in cap_map.keys()}
        tot_cap = sum(cap_map.values())
        if tot_cap <= 0 or kept_total <= 0:
            return kept_map
        # proportional split (rounding-safe)
        remaining = kept_total
        classes = list(cap_map.keys())
        for idx, c in enumerate(classes):
            if idx < len(classes) - 1:
                share = int(round(kept_total * (cap_map[c] / float(tot_cap))))
                share = min(share, remaining)
            else:
                share = remaining
            kept_map[c] += share
            gen_counter[int(c)] += share
            quota[int(c)] = max(0, int(quota[int(c)]) - share)
            remaining -= share
        return kept_map

    @torch.no_grad()
    def _generate_for_cap_map(cap_map: Dict[int, int], margin_gate):
        """
        Generate for a given {class: requested_amount} map, in chunks.
        Critically, we now decrement the remaining request per class by the **KEPT**
        amount (post margin gate), not by the requested amount.
        """
        # Make an editable copy of requests
        remaining_map = {int(c): int(v) for c, v in cap_map.items() if int(v) > 0}
        if not remaining_map:
            return 0

        kept_total_all = 0
        no_progress_rounds = 0  # guard against infinite loops if gate keeps rejecting

        while remaining_map:
            # Build a chunk up to max_gen_chunk
            chunk_labels = []
            capacity = max_gen_chunk
            for c, need in list(remaining_map.items()):
                k = min(need, capacity)
                if k > 0:
                    chunk_labels.extend([c] * k)
                    capacity -= k
                if capacity == 0:
                    break

            if not chunk_labels:
                break  # nothing to do

            y_cap = torch.tensor(chunk_labels, device=device, dtype=torch.long)
            Zs = _orig_ddim_sample(y=y_cap, n=len(y_cap), steps=steps_infer, margin_gate=margin_gate)
            kept_chunk = int(Zs.size(0))

            # Build a map of what we asked per class in this chunk
            chunk_cap_map = {}
            for c in set(chunk_labels):
                chunk_cap_map[c] = chunk_labels.count(c)

            # Attribute kept per class and decrement quotas
            kept_map = _attribute_and_decrement(chunk_cap_map, kept_chunk)

            # Decrement remaining_map by the **kept** per class
            progress = 0
            for c, asked in chunk_cap_map.items():
                kept_c = kept_map.get(c, 0)
                if c in remaining_map:
                    remaining_map[c] = max(0, remaining_map[c] - kept_c)
                    if remaining_map[c] == 0:
                        remaining_map.pop(c, None)
                progress += kept_c

            kept_total_all += kept_chunk

            # If margin gate rejected everything in this chunk, prevent an infinite loop
            if progress == 0:
                no_progress_rounds += 1
            else:
                no_progress_rounds = 0
            if no_progress_rounds >= 3:
                print("[Diffusion][auto-balance] No progress for 3 chunks; stopping early to avoid infinite loop.")
                break

        return kept_total_all

    @torch.no_grad()
    def ddim_sample(y, n=None, steps=None, margin_gate=None):
        """
        Wrap original sampler to:
          1) (optional) **auto-balance**: immediately generate all remaining quota
             (up to the majority) the first time this is called.
          2) Otherwise, enforce per-call quotas and count kept samples.
        """
        nonlocal _auto_filled_once

        # 1) AUTO-FILL path (runs once)
        if auto_balance_to_majority and not _auto_filled_once:
            cap_map = {int(c): int(v) for c, v in quota.items() if int(v) > 0}
            kept = _generate_for_cap_map(cap_map, margin_gate)
            if kept > 0:
                print(f"[Diffusion][auto-balance] Generated and kept {kept} synthetic embeddings to reach majority.")
            _auto_filled_once = True
            # Return an empty tensor to the caller (training loop already mixes real/synth)
            return torch.empty(0, ddm.feat_dim, device=device)

        # 2) REGULAR per-call capped sampling
        y = y.to(device)
        uniq, req_cnts = torch.unique(y, return_counts=True)
        classes = uniq.tolist()
        req = {int(c): int(k) for c, k in zip(classes, req_cnts.tolist())}

        # Cap by remaining quota
        cap_map = {}
        for c in classes:
            allowed = max(0, int(quota[c]))
            take = min(req[c], allowed)
            if take > 0:
                cap_map[c] = take

        if not cap_map:
            return torch.empty(0, ddm.feat_dim, device=device)

        _ = _generate_for_cap_map(cap_map, margin_gate)
        return torch.empty(0, ddm.feat_dim, device=device)

    ddm.ddim_sample = ddim_sample
    ddm._gen_counter = gen_counter
    ddm._quota = quota
    ddm._auto_balance_to_majority = auto_balance_to_majority
    return ddm
