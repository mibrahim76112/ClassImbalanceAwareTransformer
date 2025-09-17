import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)

@torch.no_grad()
def evaluate(model, loader, device, compute_rival_gap=True):
    model.eval()
    ys, ps = [], []
    first = True
    all_gaps = []  # rival-gap per sample if cosine head

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            x, y = batch

        x = x.to(device); y = y.to(device)

        if hasattr(model, "cos_head"):
            feats = model.forward_features(x)
            logits = model.cos_head(feats, y=None, use_margin=False)
            if first: print("[EVAL] using cosine-head logits")
            if compute_rival_gap:
                # gap = cos_y - max_other (before scaling is fine for ranking)
                with torch.no_grad():
                    # remove scale for gap inspection (division ok since s>0)
                    s = getattr(model.cos_head, "s", 1.0)
                    l = logits / max(float(s), 1.0)
                    tgt = l.gather(1, y.view(-1, 1)).squeeze(1)
                    l.scatter_(1, y.view(-1, 1), -1e9)  # mask target
                    rival, _ = l.max(dim=1)
                    all_gaps.append((tgt - rival).detach().cpu())
        else:
            logits = model(x)
            if first: print("[EVAL] using linear-head logits")

        first = False
        preds = logits.argmax(dim=1)
        ys.append(y.detach().cpu()); ps.append(preds.detach().cpu())

    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    rep = classification_report(y_true, y_pred, digits=3, output_dict=True, zero_division=0)

    out = {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "report": rep,
        "y_true": y_true,
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    if all_gaps:
        gaps = torch.cat(all_gaps).numpy()
        out["rival_gap_mean"] = float(np.mean(gaps))
        out["rival_gap_p10"] = float(np.percentile(gaps, 10))
        out["rival_gap_p50"] = float(np.percentile(gaps, 50))
        out["rival_gap_p90"] = float(np.percentile(gaps, 90))

    return out
