# src/plot_cli.py
import os
import argparse
import warnings
import numpy as np

from src.plots import (
    plot_class_dist_and_margins,
    plot_per_class_bars,
    plot_embedding_with_centers,
    plot_center_similarity,
    plot_inter_intra_distributions,
    plot_learning_curves,
    plot_reliability_diagram,
)

def _p(path): 
    return os.path.abspath(path)

def _exists(p): 
    return (p is not None) and os.path.exists(p)

def main():
    ap = argparse.ArgumentParser(description="Paper plots CLI (no-TTA)")
    ap.add_argument("--results-dir", default="results",
                    help="Folder where train.py saved artifacts/figures")
    ap.add_argument(
        "--plots", nargs="+",
        default=["confmat", "calib", "classdist", "centers", "embed", "interintra", "curves"],
        choices=["confmat","perclass","calib","classdist","centers","embed","interintra","curves","all"],
        help="Which plots to make"
    )
    ap.add_argument("--annotate-threshold", type=float, default=0.02,
                    help="Only annotate CM cells with value >= this threshold (row-normalized)")
    ap.add_argument("--suppress-warnings", action="store_true",
                    help="Silence matplotlib/torch warnings")
    ap.add_argument("--umap", action="store_true",
                    help="Force UMAP (falls back to t-SNE if UMAP not installed)")
    ap.add_argument("--baseline-report", default=None,
                    help="Path to baseline npz with arrays y_true_base,y_pred_base for per-class bars")
    args = ap.parse_args()

    if args.suppress_warnings:
        warnings.filterwarnings("ignore")

    rd = _p(args.results_dir)
    os.makedirs(rd, exist_ok=True)

    # Expected artifacts from train.py
    y_train_path     = os.path.join(rd, "y_train.npy")            # optional (class-dist)
    y_true_path      = os.path.join(rd, "test_y.npy")
    y_pred_reg_path  = os.path.join(rd, "test_pred_reg.npy")
    logits_reg_path  = os.path.join(rd, "test_logits_reg.npy")    # optional (calibration)
    centers_path     = os.path.join(rd, "centers.npy")            # optional (center plots)
    feats_path       = os.path.join(rd, "test_feats.npy")         # optional (embedding)

    # Load what we can
    y_true = np.load(y_true_path) if _exists(y_true_path) else None
    y_pred_reg = np.load(y_pred_reg_path) if _exists(y_pred_reg_path) else None
    logits_reg = np.load(logits_reg_path) if _exists(logits_reg_path) else None
    centers = np.load(centers_path) if _exists(centers_path) else None
    feats   = np.load(feats_path) if _exists(feats_path) else None
    y_train = np.load(y_train_path) if _exists(y_train_path) else None

    wanted = set(args.plots)
    if "all" in wanted:
        wanted = {"confmat","calib","classdist","centers","embed","interintra","curves"}

    # 1) Class distribution vs margins
    if "classdist" in wanted:
        # Dummy model to let function run even without margins
        class _Dummy: pass
        dummy = _Dummy(); dummy.cos_head = _Dummy()
        dummy.cos_head.per_class_margin = None
        if y_train is not None:
            plot_class_dist_and_margins(
                y_train, dummy, save_path=os.path.join(rd, "class_dist_margins.png")
            )
            print("[OK] class_dist_margins.png")
        else:
            print("[SKIP] classdist: y_train.npy not found")

    # 3) Per-class bars (Baseline vs Ours) -> needs a baseline npz
    if "perclass" in wanted:
        if args.baseline_report and os.path.exists(args.baseline_report):
            d = np.load(args.baseline_report, allow_pickle=True)
            y_true_base = d["y_true_base"]; y_pred_base = d["y_pred_base"]
            if (y_true is None) or (y_pred_reg is None):
                print("[SKIP] perclass: need y_true / y_pred_reg from current model")
            else:
                plot_per_class_bars(
                    y_true_base, y_pred_base, y_true, y_pred_reg,
                    metric="recall", save_path=os.path.join(rd, "perclass_recall.png")
                )
                print("[OK] perclass_recall.png")
        else:
            print("[SKIP] perclass: provide --baseline-report path with y_true_base,y_pred_base")

    # 4) Embedding with centers
    if "embed" in wanted:
        if (feats is not None) and (y_true is not None) and (centers is not None):
            method = "umap" if args.umap else "tsne"
            plot_embedding_with_centers(
                feats, y_true, centers=centers, method=method,
                save_path=os.path.join(rd, "embed_centers.png")
            )
            print("[OK] embed_centers.png")
        else:
            print("[SKIP] embed: need test_feats.npy and centers.npy (and test_y.npy)")

    # 5) Center similarity heatmap
    if "centers" in wanted:
        if centers is not None:
            plot_center_similarity(
                centers, save_path=os.path.join(rd, "center_similarity.png")
            )
            print("[OK] center_similarity.png")
        else:
            print("[SKIP] centers: centers.npy not found")

    # 6) Inter-/intra-class distance distributions
    if "interintra" in wanted:
        if (feats is not None) and (y_true is not None) and (centers is not None):
            plot_inter_intra_distributions(
                feats, y_true, centers,
                save_path=os.path.join(rd, "inter_intra_distributions.png")
            )
            print("[OK] inter_intra_distributions.png")
        else:
            print("[SKIP] interintra: need test_feats.npy, centers.npy, test_y.npy")

    # 7) Learning curves (if you saved history.npz)
    if "curves" in wanted:
        hist_path = os.path.join(rd, "history.npz")
        if os.path.exists(hist_path):
            h = np.load(hist_path, allow_pickle=True)
            history = {k: h[k].tolist() for k in h.files}
            plot_learning_curves(history, save_path=os.path.join(rd, "learning_curves.png"))
            print("[OK] learning_curves.png")
        else:
            print("[SKIP] curves: history.npz not found")

    # 8) Calibration (reliability + ECE)
    if "calib" in wanted:
        if (logits_reg is not None) and (y_true is not None):
            plot_reliability_diagram(
                logits_reg, y_true, save_path=os.path.join(rd, "reliability.png"), n_bins=15
            )
            print("[OK] reliability.png")
        else:
            print("[SKIP] calib: need test_logits_reg.npy and test_y.npy")

if __name__ == "__main__":
    main()
