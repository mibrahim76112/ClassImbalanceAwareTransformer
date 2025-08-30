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

@torch.no_grad()
def evaluate_tta(model, loader, device, K=8):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        feats0 = model.forward_features(x)
        logits0 = model.cos_head(feats0, y=None, use_margin=False)
        logit_sum = F.log_softmax(logits0, dim=1)
        for _ in range(K):
            xw = x + torch.randn_like(x) * 0.0007
            s  = torch.empty(x.size(0), 1, 1, device=device).uniform_(0.99, 1.01)
            xw = xw * s
            feats = model.forward_features(xw)
            logits = model.cos_head(feats, y=None, use_margin=False)
            logit_sum += F.log_softmax(logits, dim=1)
        preds = logit_sum.argmax(dim=1)
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
