import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluation that supports two cases:
      1) Baseline: model(x) returns logits from the internal linear classifier.
      2) Cosine head: use model.forward_features(x) -> cos_head(...) for logits.
    """
    model.eval()
    ys, ps = [], []
    for batch in loader:
        # robust unpack (works if dataset returns extra views for contrastive setups)
        if isinstance(batch, (list, tuple)):
            x, y = batch[0], batch[1]
        else:
            x, y = batch

        x = x.to(device); y = y.to(device)

        if hasattr(model, "cos_head"):
            feats = model.forward_features(x)
            logits = model.cos_head(feats, y=None, use_margin=False)
        else:
            logits = model(x)  # baseline: linear classifier inside the model

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
