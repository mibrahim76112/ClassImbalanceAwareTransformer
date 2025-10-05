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
    ap.add_argument("--results-dir", default="results", help="folder where train.py saved artifacts")
    ap.add_argument(
        "--plots", nargs="+",
        default=["classdist","centers","embed","interintra","celoss","valmetrics"],
        choices=[
            "perclass","classdist","centers","embed","embed3","interintra","celoss","valmetrics","all",
            "diffbalance","diffmix"
        ],
        help="Which plots to make"
    )
    ap.add_argument("--annotate-threshold", type=float, default=0.02)
    ap.add_argument("--suppress-warnings", action="store_true")
    ap.add_argument("--umap", action="store_true",
                    help="Use UMAP for the regular 'embed' plot. (embed3 is always t-SNE)")
    ap.add_argument("--baseline-report", default=None)

    # generated embeddings inputs
    ap.add_argument("--gen-all-path", default=None)
    ap.add_argument("--fault", type=int, default=6)
    ap.add_argument("--normal-label", type=int, default=0)
    ap.add_argument("--max-per-group", type=int, default=650)

    # NEW: use TRAIN set directly (no big npy)
    ap.add_argument("--use-train", action="store_true",
                    help="For embed3, compute features from a small TRAIN subset on-the-fly")
    ap.add_argument("--config", type=str, default=None,
                    help="Path to config.yaml (required if --use-train)")

    args = ap.parse_args()
    if args.suppress_warnings:
        warnings.filterwarnings("ignore")

    rd = _p(args.results_dir); os.makedirs(rd, exist_ok=True)

    # standard artifacts (only used when not --use-train)
    y_train_path = os.path.join(rd, "y_train.npy")
    y_true_path  = os.path.join(rd, "test_y.npy")
    y_pred_path  = os.path.join(rd, "test_pred_reg.npy")
    feats_path   = os.path.join(rd, "test_feats.npy")
    centers_path = os.path.join(rd, "centers.npy")
    hist_path    = os.path.join(rd, "history.npz")
    train_counts_path = os.path.join(rd, "train_counts.npy")
    diff_synth_path   = os.path.join(rd, "diffusion_synth_counts.npy")

    gen_all_path = args.gen_all_path or os.path.join(rd, "gen_all.npy")

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

    if "embed3" in wanted:
        if args.use-train:
            try:
                import yaml, torch
                from pathlib import Path
                from src.data import load_sampled_data
                from src.datasets import PlainTSDataset
                from src.model import SelfGatedHierarchicalTransformerEncoder, CosineMarginClassifier
                from src.plots import get_features

                if not args.config or not os.path.exists(args.config):
                    print("[SKIP] embed3 --use-train: need --config path to config.yaml")
                else:
                    with open(args.config, "r") as f:
                        cfg = yaml.safe_load(f)

                    ff_path = cfg["dataset"]["ff_path"]
                    ft_path = cfg["dataset"]["ft_path"]
                    window_size = cfg["data_windowing"]["window_size"]
                    stride      = cfg["data_windowing"]["stride"]
                    post_fault_start = cfg["data_windowing"]["post_fault_start"]
                    train_runs = range(cfg["data_windowing"]["train_runs_start"],
                                       cfg["data_windowing"]["train_runs_end"])

                    (X_train, y_train_arr, _) , _ = load_sampled_data(
                        window_size=window_size, stride=stride,
                        ff_path=ff_path, ft_path=ft_path,
                        post_fault_start=post_fault_start,
                        train_runs=train_runs, test_runs=[]
                    )

                    rng = np.random.default_rng(0)
                    def take_per_class(X, y, per_cls):
                        idx_all = []
                        for c in np.unique(y):
                            idx = np.where(y == c)[0]
                            if len(idx) > per_cls:
                                idx = rng.choice(idx, size=per_cls, replace=False)
                            idx_all.append(idx)
                        return np.concatenate(idx_all, axis=0)

                    # small subset for speed/memory
                    sel_idx = take_per_class(X_train, y_train_arr.astype(int), per_cls=min(800, args.max_per_group))
                    X_small, y_small = X_train[sel_idx], y_train_arr[sel_idx]

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    num_classes = int(y_train_arr.max()) + 1
                    input_dim = X_small.shape[2]

                    model = SelfGatedHierarchicalTransformerEncoder(
                        input_dim=input_dim, num_classes=num_classes
                    ).to(device)

                    best_path = os.path.join(rd, "best_state_dict.pt")
                    if not os.path.exists(best_path):
                        print("[SKIP] embed3 --use-train: best_state_dict.pt not found")
                    else:
                        dummy_loader = torch.utils.data.DataLoader(
                            PlainTSDataset(X_small[:32], y_small[:32]),
                            batch_size=32, shuffle=False, num_workers=0, pin_memory=True
                        )
                        with torch.no_grad():
                            xb, yb = next(iter(dummy_loader))
                            xb = xb.to(device)
                            feat_dim = model.forward_features(xb).shape[-1]
                        model.cos_head = CosineMarginClassifier(
                            feat_dim=feat_dim, num_classes=num_classes,
                            s=float(cfg["model"]["s"]), m=float(cfg["model"]["m"]),
                            margin_type=str(cfg["model"]["margin_type"])
                        ).to(device)
                        sd = torch.load(best_path, map_location=device)
                        model.load_state_dict(sd)

                        dl = torch.utils.data.DataLoader(
                            PlainTSDataset(X_small, y_small),
                            batch_size=256, shuffle=False, num_workers=0, pin_memory=True
                        )
                        feats_train, y_train_small = get_features(model, dl, device)

                        # load generated embeddings (if available), prefer unnormalized
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
                                d = np.load(gen_sel_un, allow_pickle=True)
                                if isinstance(d, np.ndarray) and d.dtype == object: d = d.item()
                                Z_gen = d.get(str(args.fault), None)
                        if Z_gen is None:
                            gen_sel = os.path.join(rd, "gen_selected.npy")
                            if _exists(gen_sel):
                                d = np.load(gen_sel, allow_pickle=True)
                                if isinstance(d, np.ndarray) and d.dtype == object: d = d.item()
                                Z_gen = d.get(str(args.fault), None)
                        if Z_gen is None:
                            if _exists(gen_all_path):
                                d = np.load(gen_all_path, allow_pickle=True)
                                if isinstance(d, np.ndarray) and d.dtype == object: d = d.item()
                                Z_gen = d.get(str(args.fault), None)

                        if Z_gen is None or (hasattr(Z_gen, "size") and Z_gen.size == 0):
                            print(f"[WARN] embed3 --use-train: no generated vectors for fault {args.fault}; plotting without Generated.")
                            Z_gen = np.empty((0, feats_train.shape[1]), dtype=feats_train.dtype)

                        plot_tsne_normal_fault6_generated(
                            feats_real=feats_train,
                            y_real=y_train_small.astype(int),
                            gen_fault6_feats=Z_gen,
                            normal_label=args.normal_label,
                            fault6_label=args.fault,
                            save_path=os.path.join(rd, f"tsne_norm_f{args.fault}_gen_train.png"),
                            max_per_group=args.max_per_group,
                            seed=0,
                            normalize="both",
                        )
                        print(f"[OK] tsne_norm_f{args.fault}_gen_train.png")
            except Exception as e:
                print(f"[SKIP] embed3 --use-train failed: {e}")

        else:
            if feats is not None and y_true is not None:
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
                    Z_gen = np.empty((0, feats.shape[1]), dtype=feats.dtype)

                plot_tsne_normal_fault6_generated(
                    feats_real=feats,
                    y_real=y_true.astype(int),
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
                print("[SKIP] embed3: need test_feats.npy and test_y.npy (or pass --use-train)")

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
        if _exists(hist_path):
            h = np.load(hist_path, allow_pickle=True)
            history = {k: h[k].tolist() for k in h.files}
            plot_ce_loss(history, save_path=os.path.join(rd, "ce_loss.png"))
            print("[OK] ce_loss.png")
        else:
            print("[SKIP] celoss: history.npz not found")

    if "valmetrics" in wanted:
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
            plot_real_vs_diffusion_counts(train_counts, synth_counts,
                                          save_path=os.path.join(rd, "diffusion_balance_counts.png"))
            print("[OK] diffusion_balance_counts.png")
        else:
            print("[SKIP] diffbalance: need train_counts.npy and diffusion_synth_counts.npy")

    if "diffmix" in wanted:
        if _exists(train_counts_path) and _exists(diff_synth_path):
            train_counts = np.load(train_counts_path)
            synth_counts = np.load(diff_synth_path)
            plot_effective_training_distribution(train_counts, synth_counts,
                                                 save_path=os.path.join(rd, "diffusion_effective_distribution.png"))
            print("[OK] diffusion_effective_distribution.png")
        else:
            print("[SKIP] diffmix: need train_counts.npy and diffusion_synth_counts.npy")

if __name__ == "__main__":
    main()
