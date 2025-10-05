import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    accuracy_score, balanced_accuracy_score, f1_score
)
from sklearn.manifold import TSNE

@torch.no_grad()
def get_preds_and_logits(model, loader, device):
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

def plot_real_vs_diffusion_counts(train_counts, synth_counts,
                                  save_path="results/diffusion_balance_counts.png"):
    _ensure_dir(os.path.dirname(save_path) or ".")
    import numpy as np
    C = max(len(train_counts), len(synth_counts))
    real = np.array(train_counts, dtype=float)
    synth = np.array(synth_counts, dtype=float)
    if len(real) < C:  real = np.pad(real, (0, C - len(real)))
    if len(synth) < C: synth = np.pad(synth, (0, C - len(synth)))
    after = real + synth

    x = np.arange(C)
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - w/2, real,  width=w, label="Real (pre-diffusion)", alpha=0.85)
    ax.bar(x + w/2, after, width=w, label="After (real + diffusion)", alpha=0.85)
    ax.set_xlabel("Class")
    ax.set_ylabel("Counts")
    ax.set_xticks(x)
    ax.set_title("Per-class counts: before vs. after diffusion balancing")
    ax.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig)

def plot_effective_training_distribution(train_counts, synth_counts,
                                         save_path="results/diffusion_effective_distribution.png"):
    _ensure_dir(os.path.dirname(save_path) or ".")
    import numpy as np
    real = np.array(train_counts, dtype=float)
    synth = np.array(synth_counts, dtype=float)
    C = max(len(real), len(synth))
    if len(real) < C:  real = np.pad(real, (0, C - len(real)))
    if len(synth) < C: synth = np.pad(synth, (0, C - len(synth)))
    post = real + synth

    pre_pdf  = real / max(1.0, real.sum())
    post_pdf = post / max(1.0, post.sum())

    x = np.arange(C)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x, real,  label="Real", bottom=0.0, alpha=0.85)
    ax.bar(x, synth, label="Synthetic (diffusion)", bottom=real, alpha=0.85)

    ax2 = ax.twinx()
    ax2.plot(x, pre_pdf,  marker="o", linewidth=1.75, label="Pre (normalized)")
    ax2.plot(x, post_pdf, marker="s", linewidth=1.75, label="Post (normalized)")

    ax.set_xlabel("Class"); ax.set_ylabel("Counts")
    ax2.set_ylabel("Fraction")
    ax.set_title("Effective training mix: real vs diffusion per class\n(+ normalized pre/post overlays)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig)

def plot_tsne_normal_fault6_generated(
    feats_real,
    y_real,
    gen_fault6_feats=None,
    normal_label=0,
    fault6_label=6,
    save_path="results/tsne_norm_f6_gen.png",
    max_per_group=650,
    seed=0,
    normalize="both",
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    rng = np.random.default_rng(seed)

    idx_norm = np.where(y_real == normal_label)[0]
    idx_f    = np.where(y_real == fault6_label)[0]

    def _pick(idx, K):
        return rng.choice(idx, size=min(K, len(idx)), replace=False) if len(idx) > K else idx

    idx_norm = _pick(idx_norm, max_per_group)
    idx_f    = _pick(idx_f,    max_per_group)

    X_norm = feats_real[idx_norm]
    X_f    = feats_real[idx_f]
    X_gen  = np.asarray(gen_fault6_feats) if gen_fault6_feats is not None else np.empty((0, X_norm.shape[1]))

    if normalize == "both":
        def _l2n(a, eps=1e-12):
            n = np.linalg.norm(a, axis=1, keepdims=True)
            return a / np.clip(n, eps, None)
        if X_norm.size: X_norm = _l2n(X_norm)
        if X_f.size:    X_f    = _l2n(X_f)
        if X_gen.size:  X_gen  = _l2n(X_gen)

    X_all = np.concatenate([X_norm, X_f, X_gen], axis=0)
    Z_all = TSNE(n_components=2, learning_rate="auto", init="random",
                 perplexity=35, random_state=seed).fit_transform(X_all)

    n_norm = len(X_norm); n_f = len(X_f)
    a = 0; b = a + n_norm; c = b + n_f
    Z_norm, Z_f, Z_gen = Z_all[a:b], Z_all[b:c], Z_all[c:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(Z_norm[:,0], Z_norm[:,1], s=10, c="#1f77b4", label="Normal")
    ax.scatter(Z_f[:,0],    Z_f[:,1],    s=18, c="#ff7f0e", label=f"Fault {fault6_label}")
    if len(Z_gen):
        ax.scatter(Z_gen[:,0], Z_gen[:,1], s=18, c="#2ca02c", label=f"Fault {fault6_label} Generated")
    ax.legend(frameon=True, loc="best")
    ax.set_title("t-SNE: Normal vs Fault vs Generated")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig)

def plot_tsne_triplet(
    feats, y, gen, fault_id,
    save_path="results/embed_fault_triplet.png",
    normalize="both",
    max_per_class=600,
    seed=0,
    perplexity=35
):
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    def _l2n(a, eps=1e-12):
        n = np.linalg.norm(a, axis=1, keepdims=True)
        return a / np.clip(n, eps, None)

    rng = np.random.default_rng(seed)

    idx_norm = np.where(y == 0)[0]
    idx_fk   = np.where(y == int(fault_id))[0]
    if len(idx_norm) == 0 or len(idx_fk) == 0 or gen is None or len(gen) == 0:
        raise ValueError("Missing data for plot_tsne_triplet (normal/fault/gen).")

    if len(idx_norm) > max_per_class:
        idx_norm = rng.choice(idx_norm, size=max_per_class, replace=False)
    if len(idx_fk) > max_per_class:
        idx_fk = rng.choice(idx_fk, size=max_per_class, replace=False)
    if len(gen) > max_per_class:
        gen = gen[rng.choice(np.arange(len(gen)), size=max_per_class, replace=False)]

    Xn = feats[idx_norm]
    Xk = feats[idx_fk]
    Gk = gen

    if normalize == "both":
        Xn = _l2n(Xn); Xk = _l2n(Xk); Gk = _l2n(Gk)

    X_all = np.vstack([Xn, Xk, Gk])
    y_all = np.hstack([
        np.zeros(len(Xn), dtype=int),
        np.ones(len(Xk), dtype=int),
        np.full(len(Gk), 2, dtype=int)
    ])

    Z = TSNE(n_components=2, learning_rate="auto", init="random",
             perplexity=perplexity, random_state=seed).fit_transform(X_all)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(Z[y_all==0,0], Z[y_all==0,1], s=14, label="Normal", alpha=0.85)
    ax.scatter(Z[y_all==1,0], Z[y_all==1,1], s=14, label=f"Fault {fault_id}", alpha=0.85)
    ax.scatter(Z[y_all==2,0], Z[y_all==2,1], s=14, label=f"Fault {fault_id} Generated", alpha=0.85)
    ax.set_title("t-SNE: Normal vs Fault vs Generated")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
