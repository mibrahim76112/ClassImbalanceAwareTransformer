import os
import argparse
import warnings
import numpy as np
import torch
import yaml

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
from src.data import load_sampled_data
from src.datasets import PlainTSDataset
from src.model import SelfGatedHierarchicalTransformerEncoder


def _p(path): return os.path.abspath(path)
def _exists(p): return p is not None and os.path.exists(p)


def main():
    ap = argparse.ArgumentParser(description="Paper plots CLI (no-reliability)")

    ap.add_argument("--normalize", default="zscore", choices=["none","unit","zscore","both"])
    ap.add_argument("--normal-cap", type=int, default=350)
    ap.add_argument("--fault-real-cap", type=int, default=20)
    ap.add_argument("--perplexity", type=int, default=35)

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
    ap.add_argument("--use-train", action="store_true")
    ap.add_argument("--config", type=str, default="config.yaml")

    args = ap.parse_args()
    if args.suppress_warnings:
        warnings.filterwarnings("ignore")

    rd = _p(args.results_dir)
    os.makedirs(rd, exist_ok=True)

    y_train_path = os.path.join(rd, "y_train.npy")
    y_true_path = os.path.join(rd, "test_y.npy")
    y_pred_path = os.path.join(rd, "test_pred_reg.npy")
    feats_path = os.path.join(rd, "test_feats.npy")
    centers_path = os.path.join(rd, "centers.npy")
    hist_path = os.path.join(rd, "history.npz")
    train_counts_path = os.path.join(rd, "train_counts.npy")
    diff_synth_path = os.path.join(rd, "diffusion_synth_counts.npy")

    gen_all_path = args.gen_all_path or os.path.join(rd, "gen_all.npy")

    y_train = np.load(y_train_path) if _exists(y_train_path) else None
    y_true = np.load(y_true_path) if _exists(y_true_path) else None
    y_pred = np.load(y_pred_path) if _exists(y_pred_path) else None
    feats = np.load(feats_path) if _exists(feats_path) else None
    centers = np.load(centers_path) if _exists(centers_path) else None

    feats_train = None
    y_train_small = None
    if args.use_train:
        with open(args.config, "r") as f:
            _cfg = yaml.safe_load(f)
        ws  = _cfg["data_windowing"]["window_size"]
        st  = _cfg["data_windowing"]["stride"]
        ff  = _cfg["dataset"]["ff_path"]
        ft  = _cfg["dataset"]["ft_path"]
        pfs = _cfg["data_windowing"]["post_fault_start"]
        tr_start = _cfg["data_windowing"]["train_runs_start"]
        tr_end   = _cfg["data_windowing"]["train_runs_end"]
        train_runs = range(tr_start, tr_end)

        (X_train_full, y_train_all, _), _ = load_sampled_data(
            window_size=ws, stride=st,
            ff_path=ff, ft_path=ft,
            post_fault_start=pfs,
            train_runs=train_runs,
            test_runs=train_runs
        )

        rng = np.random.default_rng(0)
        take_per_class = min(600, args.max_per_group)
        idx_sel = []
        for c in np.unique(y_train_all):
            idx_c = np.where(y_train_all == c)[0]
            k = min(take_per_class, len(idx_c))
            idx_sel.append(rng.choice(idx_c, size=k, replace=False) if len(idx_c) > k else idx_c)
        idx_sel = np.concatenate(idx_sel)
        X_small = X_train_full[idx_sel]
        y_train_small = y_train_all[idx_sel].astype(int)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SelfGatedHierarchicalTransformerEncoder(
            input_dim=X_small.shape[2],
            num_classes=int(y_train_all.max()) + 1
        ).to(device)
        best_path = os.path.join(rd, "best_state_dict.pt")
        if os.path.exists(best_path):
            state = torch.load(best_path, map_location=device)
            model.load_state_dict(state, strict=False)
        model.eval()
        ds = PlainTSDataset(X_small, y_train_small)
        dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
        feats_list = []
        with torch.no_grad():
            for xb, _ in dl:
                xb = xb.to(device)
                z = model.forward_features(xb).detach().cpu().numpy()
                feats_list.append(z)
        feats_train = np.concatenate(feats_list, axis=0)


    wanted = set(args.plots)
    if "all" in wanted:
        wanted = {"classdist", "centers", "embed", "interintra", "celoss", "valmetrics"}

    if "classdist" in wanted:
        class _Dummy: ...
        dummy = _Dummy()
        dummy.cos_head = _Dummy()
        dummy.cos_head.per_class_margin = None
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
        if (feats is not None and y_true is not None) or (args.use_train and feats_train is not None and y_train_small is not None):
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
                if feats is not None:
                    Z_gen = np.empty((0, feats.shape[1]), dtype=feats.dtype)
                elif feats_train is not None:
                    Z_gen = np.empty((0, feats_train.shape[1]), dtype=feats_train.dtype)

            if args.use_train and (feats_train is not None) and (y_train_small is not None):
                feats_use = feats_train
                y_use = y_train_small
            else:
                feats_use = feats
                y_use = y_true

            plot_tsne_normal_fault6_generated(
                            feats_real=feats_use,
                            y_real=y_use.astype(int),
                            gen_fault6_feats=Z_gen,
                            normal_label=args.normal_label,
                            fault6_label=args.fault,
                            save_path=os.path.join(rd, f"tsne_norm_f{args.fault}_gen.png"),
                            max_per_group=args.max_per_group,
                            seed=0,
                            normalize=args.normalize,
                            normal_cap=args.normal_cap,
                            fault_real_cap=args.fault_real_cap,
                            perplexity=args.perplexity,
                        )

            print(f"[OK] tsne_norm_f{args.fault}_gen.png")
        else:
            print("[SKIP] embed3: need features/labels from test or --use-train subset")

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
