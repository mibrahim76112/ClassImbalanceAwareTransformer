# src/plots.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, balanced_accuracy_score, f1_score
)

@torch.no_grad()
def get_preds_and_logits(model, loader, device):
    """Return y_true, y_pred, logits (CPU)."""
    model.eval()
    ys, ps, ls = [], [], []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        feats  = model.forward_features(x)
        logits = model.cos_head(feats, y=None, use_margin=False)
        preds  = logits.argmax(dim=1)
        ys.append(y.cpu()); ps.append(preds.cpu()); ls.append(logits.detach().cpu())
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    logits_all = torch.cat(ls).numpy()
    return y_true, y_pred, logits_all

@torch.no_grad()
def get_features(model, loader, device):
    """Return (feats, y) from forward_features()."""
    model.eval()
    feats_all, ys = [], []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        feats_all.append(model.forward_features(x).detach().cpu())
        ys.append(y.cpu())
    feats = torch.cat(feats_all).numpy()
    y = torch.cat(ys).numpy()
    return feats, y

def _ensure_dir(p="results"):
    os.makedirs(p, exist_ok=True)
    return p

def plot_class_dist_and_margins(y_train, model,
                                save_path="results/class_dist_margins.png",
                                highlight=(0, 9, 15)):
    _ensure_dir(os.path.dirname(save_path) or ".")
    labels, counts = np.unique(y_train.astype(int), return_counts=True)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(labels, counts, alpha=0.85)
    ax1.set_xlabel("Class"); ax1.set_ylabel("Train count"); ax1.set_xticks(labels)

    margins = None
    if hasattr(model, "cos_head"):
        if getattr(model.cos_head, "per_class_margin", None) is not None:
            margins = model.cos_head.per_class_margin.detach().cpu().numpy()
        elif hasattr(model.cos_head, "m"):
            margins = np.full(len(labels), float(model.cos_head.m))

    if margins is not None:
        ax2 = ax1.twinx()
        ax2.plot(labels, margins, marker="o", linewidth=2)
        ax2.set_ylabel("Per-class margin $m_c$")
        for c in highlight:
            if c in labels:
                i = int(np.where(labels == c)[0])
                ax1.bar(c, counts[i], color="#d62728", alpha=0.55)
                ax2.plot(c, margins[c], marker="o", color="#d62728")

    plt.title("Class counts and per-class margins")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig)


def per_class_report(y_true, y_pred):
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    classes = sorted([int(k) for k in rep.keys() if k.isdigit()])
    f1  = np.array([rep[str(c)]["f1-score"]  for c in classes])
    rec = np.array([rep[str(c)]["recall"]    for c in classes])
    pre = np.array([rep[str(c)]["precision"] for c in classes])
    return np.array(classes), pre, rec, f1

def plot_per_class_bars(y_true_base, y_pred_base, y_true_ours, y_pred_ours,
                        metric="recall", save_path="results/per_class_bars.png",
                        highlight=(0, 9, 15)):
    _ensure_dir(os.path.dirname(save_path) or ".")
    classes_b, pre_b, rec_b, f1_b = per_class_report(y_true_base, y_pred_base)
    classes_o, pre_o, rec_o, f1_o = per_class_report(y_true_ours, y_pred_ours)
    assert (classes_b == classes_o).all(), "Class sets differ!"

    if metric.lower() == "recall":
        mb, mo, title = rec_b, rec_o, "Per-class Recall"
    elif metric.lower() == "precision":
        mb, mo, title = pre_b, pre_o, "Per-class Precision"
    else:
        mb, mo, title = f1_b, f1_o, "Per-class F1"

    x = np.arange(len(classes_b)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - w/2, mb, width=w, label="Baseline", alpha=0.85)
    ax.bar(x + w/2, mo, width=w, label="Ours",     alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(classes_b, rotation=90)
    ax.set_ylim(0.0, 1.05); ax.set_ylabel(metric.capitalize())
    ax.set_title(f"{title}: Baseline vs Ours"); ax.legend()

    for c in highlight:
        if c in classes_b:
            i = int(np.where(classes_b == c)[0])
            ax.axvspan(i-0.5, i+0.5, color="#ffe6e6", zorder=-1)

    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig)

def plot_embedding(feats, y, method="umap",
                   save_path="results/embed.png",
                   sample_per_class=800, seed=0):
    _ensure_dir(os.path.dirname(save_path) or ".")
    rng = np.random.default_rng(seed)

    classes = np.unique(y)
    idx_sel = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        idx_sel.append(rng.choice(idx_c, size=min(sample_per_class, len(idx_c)), replace=False)
                       if len(idx_c) > sample_per_class else idx_c)
    idx_sel = np.concatenate(idx_sel, axis=0)

    X = feats[idx_sel]; Y = y[idx_sel]

    Z = None
    if method.lower() == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=seed)
            Z = reducer.fit_transform(X)
        except Exception:
            method = "tsne"
    if method.lower() == "tsne":
        from sklearn.manifold import TSNE
        Z = TSNE(n_components=2, learning_rate="auto", init="random",
                 perplexity=35, random_state=seed).fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=Y, s=6, cmap="tab20")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, ticks=classes)
    ax.set_title(f"{method.upper()} embedding")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig)

def plot_center_similarity(centers, save_path="results/center_similarity.png",
                           title="Center cosine similarity"):
    _ensure_dir(os.path.dirname(save_path) or ".")
    Z = centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)
    S = Z @ Z.T
    np.fill_diagonal(S, 0.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(S, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title); ax.set_xlabel("Class"); ax.set_ylabel("Class")
    ax.set_xticks(np.arange(S.shape[0])); ax.set_yticks(np.arange(S.shape[0]))
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig)

def plot_inter_intra_distributions(feats, y, centers,
                                   save_path="results/inter_intra_distributions.png"):
    _ensure_dir(os.path.dirname(save_path) or ".")
    f = feats   / np.clip(np.linalg.norm(feats,   axis=1, keepdims=True), 1e-12, None)
    c = centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)

    sim = f @ c.T
    own_sim = sim[np.arange(len(f)), y.astype(int)]
    own_dist = 1.0 - own_sim
    sim[np.arange(len(f)), y.astype(int)] = -np.inf
    nearest_other_sim = sim.max(axis=1)
    inter_dist = 1.0 - nearest_other_sim

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([own_dist, inter_dist],
               labels=["Intra (to own center)", "Inter (to nearest other)"])
    ax.set_ylabel("Cosine distance")
    ax.set_title("Intra vs. Inter-class distance distributions")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig)


def plot_ce_loss(history, save_path="results/ce_loss.png"):
    """
    Single-panel plot for cross-entropy (train) only.
    Needs: history['epoch'], history['train_ce']
    """
    _ensure_dir(os.path.dirname(save_path) or ".")
    ep = history.get("epoch", list(range(1, len(history.get("train_ce", [])) + 1)))

    fig, ax = plt.subplots(figsize=(8,4))
    if "train_ce" in history and len(history["train_ce"]):
        ax.plot(ep, history["train_ce"], label="Train CE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CE Loss")
    ax.set_title("Cross-Entropy Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_validation_metrics(history, save_path="results/val_metrics.png"):
    """
    Validation metrics only (no lambda, no SupCon).
    Plots any of the available keys: val_acc, val_bal_acc, val_macro_f1.
    """
    _ensure_dir(os.path.dirname(save_path) or ".")
    L = max(
        len(history.get("val_acc", [])),
        len(history.get("val_bal_acc", [])),
        len(history.get("val_macro_f1", [])),
        1
    )
    ep = history.get("epoch", list(range(1, L + 1)))

    fig, ax = plt.subplots(figsize=(9,4.5))
    if "val_macro_f1" in history and len(history["val_macro_f1"]):
        ax.plot(ep, history["val_macro_f1"], label="Val Macro-F1")
    if "val_bal_acc" in history and len(history["val_bal_acc"]):
        ax.plot(ep, history["val_bal_acc"], label="Val Balanced Acc.")
    if "val_acc" in history and len(history["val_acc"]):
        ax.plot(ep, history["val_acc"], label="Val Acc")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_title("Validation Metrics")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def summarize_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
