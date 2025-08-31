# src/plots.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, balanced_accuracy_score, f1_score
)

# ------------------------ #
# Core helpers (preds, feats)
# ------------------------ #

@torch.no_grad()
def get_preds_and_logits(model, loader, device):
    """
    Returns:
      y_true: (N,)
      y_pred: (N,)
      logits_all: (N, C) on CPU (float32)
    """
    model.eval()
    ys, ps, ls = [], [], []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        feats = model.forward_features(x)
        logits = model.cos_head(feats, y=None, use_margin=False)
        preds = logits.argmax(dim=1)

        ys.append(y.cpu()); ps.append(preds.cpu()); ls.append(logits.detach().cpu())
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    logits_all = torch.cat(ls).numpy()
    return y_true, y_pred, logits_all


@torch.no_grad()
def get_features(model, loader, device):
    """
    Returns:
      feats: (N, D) float32 (CPU)
      y:     (N,)   int
    """
    model.eval()
    feats_all, ys = [], []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        feats = model.forward_features(x)
        feats_all.append(feats.detach().cpu())
        ys.append(y.cpu())
    feats = torch.cat(feats_all).numpy()
    y = torch.cat(ys).numpy()
    return feats, y

def _ensure_dir(p="results"):
    os.makedirs(p, exist_ok=True)
    return p

# ------------------------ #
# 1) Class distribution vs per-class margin
# ------------------------ #

def plot_class_dist_and_margins(y_train, model, save_path="results/class_dist_margins.png", highlight=(0,9,15)):
    """
    Bars = class counts; line = per-class margin m_c (if available).
    """
    _ensure_dir(os.path.dirname(save_path) or ".")
    labels, counts = np.unique(y_train.astype(int), return_counts=True)
    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.bar(labels, counts, alpha=0.8)
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Train count")
    ax1.set_xticks(labels)

    # try to get per-class margins
    margins = None
    if hasattr(model, "cos_head"):
        if hasattr(model.cos_head, "per_class_margin") and model.cos_head.per_class_margin is not None:
            margins = model.cos_head.per_class_margin.detach().cpu().numpy()
        elif hasattr(model.cos_head, "m"):
            margins = np.full(len(labels), float(model.cos_head.m))

    if margins is not None:
        ax2 = ax1.twinx()
        ax2.plot(labels, margins, marker="o", linewidth=2)
        ax2.set_ylabel("Per-class margin (m_c)")
        for c in highlight:
            if c in labels:
                ax1.bar(c, counts[labels.tolist().index(c)], color="#d62728", alpha=0.6)
                ax2.plot(c, margins[c], marker="o", color="#d62728")

    plt.title("Class counts and per-class margins")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

# ------------------------ #
# 2) Row-normalized confusion matrix (decluttered)
# ------------------------ #

def plot_confusion_matrix_row_norm(y_true, y_pred, save_path="results/confmat_row_norm.png",
                                   annotate_threshold=0.02, cmap="Blues"):
    """
    Row-normalized CM with annotations only for cells >= annotate_threshold.
    """
    _ensure_dir(os.path.dirname(save_path) or ".")
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(np.float64) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(8,7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalized)")

    # annotate
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm_norm[i, j]
            if v >= annotate_threshold:
                ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    return cm, cm_norm

# ------------------------ #
# 3) Per-class bars (Baseline vs Ours)
# ------------------------ #

def per_class_report(y_true, y_pred):
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    # extract only class keys (digits)
    classes = sorted([int(k) for k in rep.keys() if k.isdigit()])
    f1 = np.array([rep[str(c)]["f1-score"] for c in classes])
    rec = np.array([rep[str(c)]["recall"] for c in classes])
    prec = np.array([rep[str(c)]["precision"] for c in classes])
    return np.array(classes), prec, rec, f1

def plot_per_class_bars(y_true_base, y_pred_base, y_true_ours, y_pred_ours,
                        metric="recall", save_path="results/per_class_bars.png",
                        highlight=(0,9,15)):
    """
    metric in {"recall","f1","precision"}
    """
    _ensure_dir(os.path.dirname(save_path) or ".")
    classes_b, prec_b, rec_b, f1_b = per_class_report(y_true_base, y_pred_base)
    classes_o, prec_o, rec_o, f1_o = per_class_report(y_true_ours, y_pred_ours)
    assert (classes_b == classes_o).all(), "Class sets differ!"

    if metric.lower() == "recall":
        mb, mo = rec_b, rec_o
        mname = "Per-class Recall"
    elif metric.lower() == "precision":
        mb, mo = prec_b, prec_o
        mname = "Per-class Precision"
    else:
        mb, mo = f1_b, f1_o
        mname = "Per-class F1"

    x = np.arange(len(classes_b))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11,4))
    ax.bar(x - w/2, mb, width=w, label="Baseline", alpha=0.8)
    ax.bar(x + w/2, mo, width=w, label="Ours", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(classes_b, rotation=90)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"{mname}: Baseline vs Ours")
    ax.legend()

    # highlight key classes
    for c in highlight:
        if c in classes_b:
            idx = int(np.where(classes_b == c)[0])
            ax.axvspan(idx-0.5, idx+0.5, color="#ffe6e6", zorder=-1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

# ------------------------ #
# 4) UMAP / t-SNE with prototype centers
# ------------------------ #

def plot_embedding_with_centers(feats, y, centers=None, method="umap",
                                save_path="results/embed_centers.png",
                                sample_per_class=800, seed=0):
    """
    feats: (N, D) features from model.forward_features
    y: (N,)
    centers: (C, D) or None
    method: "umap" or "tsne"
    """
    _ensure_dir(os.path.dirname(save_path) or ".")
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    idx_sel = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        if len(idx_c) > sample_per_class:
            idx_sel.append(rng.choice(idx_c, size=sample_per_class, replace=False))
        else:
            idx_sel.append(idx_c)
    idx_sel = np.concatenate(idx_sel, axis=0)

    X = feats[idx_sel]
    Y = y[idx_sel]

    # reduce
    Z = None
    if method.lower() == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=seed)
            Z = reducer.fit_transform(X)
            C2 = reducer.transform(centers) if centers is not None else None
        except Exception:
            method = "tsne"  # fallback
    if method.lower() == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=35, random_state=seed)
        Z = reducer.fit_transform(X)
        # t-SNE has no transform; project centers by nearest neighbor in high-D (approx)
        C2 = None
        if centers is not None:
            # simple linear PCA projection as a fallback for centers
            from sklearn.decomposition import PCA
            Z_pca = PCA(n_components=2, random_state=seed).fit_transform(np.vstack([X, centers]))
            C2 = Z_pca[-centers.shape[0]:]

    # plot
    fig, ax = plt.subplots(figsize=(8,6))
    sc = ax.scatter(Z[:,0], Z[:,1], c=Y, s=6, cmap="tab20")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, ticks=classes)
    if centers is not None and C2 is not None:
        ax.scatter(C2[:,0], C2[:,1], c=np.arange(len(C2)), cmap="tab20", s=120, edgecolor="k", marker="X")
        # draw lines from each center to its nearest rival center
        from sklearn.metrics.pairwise import cosine_similarity
        S = cosine_similarity(centers / np.linalg.norm(centers, axis=1, keepdims=True))
        np.fill_diagonal(S, -1.0)
        rivals = S.argmax(axis=1)
        for i, j in enumerate(rivals):
            ax.plot([C2[i,0], C2[j,0]], [C2[i,1], C2[j,1]], lw=0.8, alpha=0.4, color="gray")
    ax.set_title(f"{method.upper()} embedding with prototype centers")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

# ------------------------ #
# 5) Center–center similarity heatmap(s)
# ------------------------ #

def plot_center_similarity(centers, save_path="results/center_similarity.png", title="Center cosine similarity"):
    """
    centers: (C, D)
    """
    _ensure_dir(os.path.dirname(save_path) or ".")
    Z = centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)
    S = Z @ Z.T
    np.fill_diagonal(S, 0.0)

    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(S, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Class")
    ax.set_xticks(np.arange(S.shape[0])); ax.set_yticks(np.arange(S.shape[0]))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

# ------------------------ #
# 6) Inter-/intra-class distance distributions
# ------------------------ #

def plot_inter_intra_distributions(feats, y, centers, save_path="results/inter_intra_distributions.png"):
    """
    Cosine distance to own center vs nearest other center.
    """
    _ensure_dir(os.path.dirname(save_path) or ".")
    # normalize
    f = feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12, None)
    c = centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)

    sim_to_centers = f @ c.T    # (N, C)
    # own and nearest other
    own_sim = sim_to_centers[np.arange(len(f)), y.astype(int)]
    own_dist = 1.0 - own_sim
    sim_to_centers[np.arange(len(f)), y.astype(int)] = -np.inf
    nearest_other_sim = sim_to_centers.max(axis=1)
    inter_dist = 1.0 - nearest_other_sim

    # box/violin
    fig, ax = plt.subplots(figsize=(6,4))
    ax.boxplot([own_dist, inter_dist], labels=["Intra (to own center)", "Inter (to nearest other)"])
    ax.set_ylabel("Cosine distance")
    ax.set_title("Intra vs. Inter-class distance distributions")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

# ------------------------ #
# 7) Learning curves (needs history you collect during training)
# ------------------------ #

def plot_learning_curves(history, save_path="results/learning_curves.png"):
    """
    history: dict with optional keys:
      'epoch', 'train_ce', 'train_supcon', 'val_acc', 'val_bal_acc', 'val_macro_f1'
    """
    _ensure_dir(os.path.dirname(save_path) or ".")
    ep = history.get("epoch", list(range(1, len(history.get("train_ce", [])) + 1)))

    fig, ax = plt.subplots(figsize=(8,5))
    if "train_ce" in history: ax.plot(ep, history["train_ce"], label="Train CE")
    if "train_supcon" in history: ax.plot(ep, history["train_supcon"], label="Train SupCon")
    if "val_macro_f1" in history: ax.plot(ep, history["val_macro_f1"], label="Val Macro-F1")
    if "val_bal_acc" in history: ax.plot(ep, history["val_bal_acc"], label="Val BalAcc")
    if "val_acc" in history: ax.plot(ep, history["val_acc"], label="Val Acc")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Learning curves")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

# ------------------------ #
# 8) Calibration: reliability + ECE
# ------------------------ #

def _softmax(x, axis=1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)

def expected_calibration_error(probs, y_true, n_bins=15):
    """
    probs: (N, C) softmax probabilities
    y_true: (N,)
    ECE: sum_k (|B_k|/N) * | acc(B_k) - conf(B_k) |
    where B_k bins by max prob.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (confidences > lo) & (confidences <= hi)
        if not np.any(mask): 
            continue
        acc = (predictions[mask] == y_true[mask]).mean()
        conf = confidences[mask].mean()
        w = mask.mean()
        ece += w * abs(acc - conf)
    return float(ece)

def plot_reliability_diagram(logits, y_true, save_path="results/reliability.png", n_bins=15):
    _ensure_dir(os.path.dirname(save_path) or ".")
    probs = _softmax(logits, axis=1)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins+1)
    accs, confs = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (confidences > lo) & (confidences <= hi)
        if np.any(mask):
            accs.append(correct[mask].mean())
            confs.append(confidences[mask].mean())
        else:
            accs.append(np.nan)
            confs.append(np.nan)
    accs = np.array(accs); confs = np.array(confs)

    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot([0,1],[0,1], linestyle="--", color="gray", label="Perfect")
    ax.plot(confs, accs, marker="o", linewidth=1.5, label="Model")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ece = expected_calibration_error(probs, y_true, n_bins=n_bins)
    ax.set_title(f"Reliability diagram (ECE={ece:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

# ------------------------ #
# Utility to summarize metrics (no TTA involved)
# ------------------------ #

def summarize_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
