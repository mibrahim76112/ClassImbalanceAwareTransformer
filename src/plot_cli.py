import os, argparse, warnings, numpy as np
from src.plots import (
    plot_class_dist_and_margins, plot_per_class_bars,
    plot_embedding, plot_center_similarity, plot_inter_intra_distributions,
    plot_validation_metrics,plot_ce_loss
)

def _p(path): return os.path.abspath(path)
def _exists(p): return p is not None and os.path.exists(p)

def main():
    ap = argparse.ArgumentParser(description="Paper plots CLI (no-reliability)")
    ap.add_argument("--results-dir", default="results", help="folder where train.py saved artifacts")
    ap.add_argument(
        "--plots", nargs="+",
        default=["classdist","centers","embed","interintra","celoss","valmetrics"],
        choices=["perclass","classdist","centers","embed","interintra","celoss","valmetrics","all"],
        help="Which plots to make"
    )

    ap.add_argument("--annotate-threshold", type=float, default=0.02)
    ap.add_argument("--suppress-warnings", action="store_true")
    ap.add_argument("--umap", action="store_true")
    ap.add_argument("--baseline-report", default=None)
    args = ap.parse_args()
    if args.suppress_warnings: warnings.filterwarnings("ignore")

    rd = _p(args.results_dir); os.makedirs(rd, exist_ok=True)

    # expected artifacts saved by train.py
    y_train_path = os.path.join(rd, "y_train.npy")
    y_true_path  = os.path.join(rd, "test_y.npy")
    y_pred_path  = os.path.join(rd, "test_pred_reg.npy")
    feats_path   = os.path.join(rd, "test_feats.npy")
    centers_path = os.path.join(rd, "centers.npy")
    hist_path    = os.path.join(rd, "history.npz")

    y_train = np.load(y_train_path) if _exists(y_train_path) else None
    y_true  = np.load(y_true_path)  if _exists(y_true_path)  else None
    y_pred  = np.load(y_pred_path)  if _exists(y_pred_path)  else None
    feats   = np.load(feats_path)   if _exists(feats_path)   else None
    centers = np.load(centers_path) if _exists(centers_path) else None

    wanted = set(args.plots)
    if "all" in wanted:
        wanted = {"classdist","centers","embed","interintra","celoss","valmetrics"}


    if "classdist" in wanted:
        class _Dummy: pass
        dummy = _Dummy(); dummy.cos_head = _Dummy(); dummy.cos_head.per_class_margin = None
        if y_train is not None:
            plot_class_dist_and_margins(y_train, dummy, save_path=os.path.join(rd,"class_dist_margins.png"))
            print("[OK] class_dist_margins.png")
        else:
            print("[SKIP] classdist: y_train.npy not found")


    if "embed" in wanted:
        if feats is not None and y_true is not None:
            method = "umap" if args.umap else "tsne"
            plot_embedding(feats, y_true, method=method, save_path=os.path.join(rd,"embed.png"))
            print("[OK] embed.png")
        else:
            print("[SKIP] embed: need test_feats.npy and test_y.npy")

    if "centers" in wanted:
        if centers is not None:
            plot_center_similarity(centers, save_path=os.path.join(rd, "center_similarity.png"))
            print("[OK] center_similarity.png")
        else:
            print("[SKIP] centers: centers.npy not found")

    if "interintra" in wanted:
        if feats is not None and y_true is not None and centers is not None:
            plot_inter_intra_distributions(feats, y_true, centers,
                                           save_path=os.path.join(rd,"inter_intra_distributions.png"))
            print("[OK] inter_intra_distributions.png")
        else:
            print("[SKIP] interintra: need test_feats.npy, centers.npy, test_y.npy")

    if "celoss" in wanted:
        if os.path.exists(hist_path):
            h = np.load(hist_path, allow_pickle=True)
            history = {k: h[k].tolist() for k in h.files}
            from src.plots import plot_ce_loss
            plot_ce_loss(history, save_path=os.path.join(rd, "ce_loss.png"))
            print("[OK] ce_loss.png")
        else:
            print("[SKIP] celoss: history.npz not found")

    if "valmetrics" in wanted:
        if os.path.exists(hist_path):
            h = np.load(hist_path, allow_pickle=True)
            history = {k: h[k].tolist() for k in h.files}
            from src.plots import plot_validation_metrics
            plot_validation_metrics(history, save_path=os.path.join(rd, "val_metrics.png"))
            print("[OK] val_metrics.png")
        else:
            print("[SKIP] valmetrics: history.npz not found")

if __name__ == "__main__":
    main()
