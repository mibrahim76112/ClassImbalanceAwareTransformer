import os
import argparse
import warnings
import numpy as np

from src.plots import (
    plot_class_dist_and_margins,
    plot_per_class_bars,
    plot_embedding,
    plot_center_similarity,
    plot_inter_intra_distributions,
    plot_validation_metrics,
    plot_ce_loss,
    plot_real_vs_diffusion_counts,
    plot_effective_training_distribution,
    plot_tsne_normal_fault6_generated,
)

def _p(path): return os.path.abspath(path)
def _exists(p): return p is not None and os.path.exists(p)

def main():
    ap = argparse.ArgumentParser(description="Paper plots CLI (no-reliability)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument(
        "--plots", nargs="+",
        default=["classdist", "centers", "embed", "interintra", "celoss", "valmetrics"],
        choices=[
            "perclass", "classdist", "centers", "embed", "embed3",
            "interintra", "celoss", "valmetrics", "all",
            "diffbalance", "diffmix"
        ]
    )

    ap.add_argument("--annotate-threshold", type=float, default=0.02)
    ap.add_argument("--suppress-warnings", action="store_true")
    ap.add_argument("--umap", action="store_true")
    ap.add_argument("--baseline-report", default=None)

    ap.add_argument("--gen-all-path", default=None)
    ap.add_argument("--fault", type=int, default=6)
    ap.add_argument("--normal-label", type=int, default=0)
    ap.add_argument("--max-per-group", type=int, default=650)

    args = ap.parse_args()
    if args.suppress_warnings:
        warnings.filterwarnings("ignore")

    rd = _p(args.results_dir)
    os.makedirs(rd, exist_ok=True)

    y_train_path = os.path.join(rd, "y_train.npy")
    y_true_path  = os.path.join(rd, "test_y.npy")
    feats_path   = os.path.join(rd, "test_feats.npy")
    centers_path = os.path.join(rd, "centers.npy")
    hist_path    = os.path.join(rd, "history.npz")
    train_counts_path = os.path.join(rd, "train_counts.npy")
    diff_synth_path   = os.path.join(rd, "diffusion_synth_counts.npy")

    # train-subset (if present, use these for embed3 to show train distribution)
    train_feats_path = os.path.join(rd, "train_feats.npy")
    train_y_path     = os.path.join(rd, "train_y.npy")

    gen_all_path = args.gen_all_path or os.path.join(rd, "gen_all.npy")

    y_train = np.load(y_train_path) if _exists(y_train_path) else None
    y_true  = np.load(y_true_path)  if _exists(y_true_path)  else None
    feats   = np.load(feats_path)   if _exists(feats_path)   else None
    centers = np.load(centers_path) if _exists(centers_path) else None

    wanted = set(args.plots)
    if "all" in wanted:
        wanted = {"classdist", "centers", "embed", "interintra", "celoss", "valmetrics"}

    if "classdist" in wanted:
        class _Dummy: pass
        dummy = _Dummy(); dummy.cos_head = _Dummy(); dummy.cos_head.per_class_margin = None
        if y_train is not None:
            plot_class_dist_and_margins(y_train, dummy, save_path=os.path.join(rd, "class_dist_margins.png"))
            print("[OK] class_dist_margins.png")
        else:
            print("[SKIP] classdist: y_train.npy not found")

    if "embed" in wanted:
        if feats is not None and y_true is not None:
            method = "umap" if args.umap else "tsne"
            plot_embedding(feats, y_true, method=method, save_path=os.path.join(rd, "embed.png"))
            print("[OK] embed.png")
        else:
            print("[SKIP] embed: need test_feats.npy and test_y.npy")

    if "embed3" in wanted:
        # prefer train subset if present; otherwise use test
        feats3 = None
        y3 = None
        if _exists(train_feats_path) and _exists(train_y_path):
            feats3 = np.load(train_feats_path)
            y3     = np.load(train_y_path).astype(int)
        elif feats is not None and y_true is not None:
            feats3 = feats
            y3     = y_true.astype(int)

        if feats3 is not None and y3 is not None:
            Z_gen = None
            per_fault_unnorm = os.path.join(rd, f"gen_f{args.fault}_unnorm.npy")
            if _exists(per_fault_unnorm):
                Z_gen = np.load(per_fault_unnorm, allow_pickle=True)

            if Z_gen is None:
                per_fault_norm = os.path.join(rd, f"gen_f{args.fault}.npy")
                if _exists(per_fault_norm):
                    Z_gen = np.load(per_fault_norm, allow_pickle=True)

            if Z_gen is None:
                gen_sel_un = os.path.join(rd, "gen_selected_unnorm.npy")
                if _exists(gen_sel_un):
                    try:
                        d = np.load(gen_sel_un, allow_pickle=True)
                        if isinstance(d, np.ndarray) and d.dtype == object:
                            d = d.item()
                        Z_gen = d.get(str(args.fault), None)
                    except Exception as e:
                        print(f"[WARN] embed3: could not read gen_selected_unnorm.npy: {e}")

            if Z_gen is None:
                gen_sel = os.path.join(rd, "gen_selected.npy")
                if _exists(gen_sel):
                    try:
                        d = np.load(gen_sel, allow_pickle=True)
                        if isinstance(d, np.ndarray) and d.dtype == object:
                            d = d.item()
                        Z_gen = d.get(str(args.fault), None)
                    except Exception as e:
                        print(f"[WARN] embed3: could not read gen_selected.npy: {e}")

            if Z_gen is None:
                if _exists(gen_all_path):
                    try:
                        d = np.load(gen_all_path, allow_pickle=True)
                        if isinstance(d, np.ndarray) and d.dtype == object:
                            d = d.item()
                        Z_gen = d.get(str(args.fault), None)
                    except Exception as e:
                        print(f"[WARN] embed3: could not read gen_all.npy: {e}")

            if Z_gen is None or (hasattr(Z_gen, "size") and Z_gen.size == 0):
                print(f"[WARN] embed3: no generated vectors found for fault {args.fault}; plotting without Generated.")
                Z_gen = np.empty((0, feats3.shape[1]), dtype=feats3.dtype)

            plot_tsne_normal_fault6_generated(
                feats_real=feats3,
                y_real=y3,
                gen_fault6_feats=Z_gen,
                normal_label=args.normal_label,
                fault6_label=args.fault,
                save_path=os.path.join(rd, f"tsne_norm_f{args.fault}_gen.png"),
                max_per_group=args.max_per_group,
                seed=0,
                normalize="both",
            )
            print(f"[OK] tsne_norm_f{args.fault}_gen.png")
        else:
            print("[SKIP] embed3: need (train_feats.npy & train_y.npy) OR (test_feats.npy & test_y.npy)")

    if "centers" in wanted:
        if centers is not None:
            plot_center_similarity(centers, save_path=os.path.join(rd, "center_similarity.png"))
            print("[OK] center_similarity.png")
        else:
            print("[SKIP] centers: centers.npy not found")

    if "interintra" in wanted:
        if feats is not None and y_true is not None and centers is not None:
            plot_inter_intra_distributions(
                feats, y_true, centers,
                save_path=os.path.join(rd, "inter_intra_distributions.png")
            )
            print("[OK] inter_intra_distributions.png")
        else:
            print("[SKIP] interintra: need test_feats.npy, centers.npy, test_y.npy")

    if "celoss" in wanted:
        hist_path = os.path.join(rd, "history.npz")
        if _exists(hist_path):
            h = np.load(hist_path, allow_pickle=True)
            history = {k: h[k].tolist() for k in h.files}
            plot_ce_loss(history, save_path=os.path.join(rd, "ce_loss.png"))
            print("[OK] ce_loss.png")
        else:
            print("[SKIP] celoss: history.npz not found")

    if "valmetrics" in wanted:
        hist_path = os.path.join(rd, "history.npz")
        if _exists(hist_path):
            h = np.load(hist_path, allow_pickle=True)
            history = {k: h[k].tolist() for k in h.files}
            plot_validation_metrics(history, save_path=os.path.join(rd, "val_metrics.png"))
            print("[OK] val_metrics.png")
        else:
            print("[SKIP] valmetrics: history.npz not found")

    if "diffbalance" in wanted:
        if _exists(train_counts_path) and _exists(diff_synth_path):
            train_counts = np.load(train_counts_path)
            synth_counts = np.load(diff_synth_path)
            plot_real_vs_diffusion_counts(
                train_counts, synth_counts,
                save_path=os.path.join(rd, "diffusion_balance_counts.png")
            )
            print("[OK] diffusion_balance_counts.png")
        else:
            print("[SKIP] diffbalance: need train_counts.npy and diffusion_synth_counts.npy")

    if "diffmix" in wanted:
        if _exists(train_counts_path) and _exists(diff_synth_path):
            train_counts = np.load(train_counts_path)
            synth_counts = np.load(diff_synth_path)
            plot_effective_training_distribution(
                train_counts, synth_counts,
                save_path=os.path.join(rd, "diffusion_effective_distribution.png")
            )
            print("[OK] diffusion_effective_distribution.png")
        else:
            print("[SKIP] diffmix: need train_counts.npy and diffusion_synth_counts.npy")

if __name__ == "__main__":
    main()
