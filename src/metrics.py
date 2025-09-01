import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    first = True
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
        else:
            logits = model(x)
            if first: print("[EVAL] using linear-head logits")
        first = False

        preds = logits.argmax(dim=1)
        ys.append(y.detach().cpu()); ps.append(preds.detach().cpu())

    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    rep = classification_report(y_true, y_pred, digits=3, output_dict=True, zero_division=0)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "report": rep,
    }
