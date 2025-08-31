import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        feats = model.forward_features(x)
        logits = model.cos_head(feats, y=None, use_margin=False)
        preds = logits.argmax(dim=1)
        ys.append(y.cpu()); ps.append(preds.cpu())
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    rep = classification_report(y_true, y_pred, digits=3, output_dict=True, zero_division=0)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "report": rep,
    }
