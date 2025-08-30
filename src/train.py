import os
import json
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*enable_nested_tensor is True.*")

import argparse
from pathlib import Path
import yaml
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from .data import load_sampled_data
from .datasets import (TwoCropsTransform, TSJitter, TSScale, TSTimeMask,
                       ContrastiveTSDataset, PlainTSDataset, make_sampler)
from .model import SelfGatedHierarchicalTransformerEncoder, CosineMarginClassifier
from .proto import PrototypeCenters, CenterSeparationLoss
from .losses import train_one_epoch
from .metrics import evaluate, evaluate_tta



def pretty_print_metrics(tag, m):
    # m: dict from evaluate()
    acc = m.get("acc", None)
    bal = m.get("bal_acc", None)
    f1  = m.get("macro_f1", None)
    print(f"{tag}: Acc={acc:.3f} | BalAcc={float(bal):.3f} | MacroF1={f1:.3f}")

def parse_args():
    p = argparse.ArgumentParser(description="TEP classifier training (config-driven)")
    p.add_argument("--config", type=str, default=str(Path(__file__).parent.parent / "config.yaml"),
                   help="Path to config.yaml")
    # CLI overrides (optional)
    p.add_argument("--ff-path", type=str, default=None)
    p.add_argument("--ft-path", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--train-batch", type=int, default=None)
    p.add_argument("--val-batch", type=int, default=None)
    p.add_argument("--test-batch", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--post-fault-start", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()

def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()

    # ---- Load YAML config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # ---- Resolve params (CLI overrides > config)
    seed = args.seed or cfg["training"]["seed"]
    set_seed(seed)

    ff_path = args.ff_path or cfg["dataset"]["ff_path"]
    ft_path = args.ft_path or cfg["dataset"]["ft_path"]

    window_size = args.window_size or cfg["data_windowing"]["window_size"]
    stride      = args.stride or cfg["data_windowing"]["stride"]
    post_fault_start = args.post_fault_start or cfg["data_windowing"]["post_fault_start"]

    train_runs = range(cfg["data_windowing"]["train_runs_start"], cfg["data_windowing"]["train_runs_end"])
    test_runs  = range(cfg["data_windowing"]["test_runs_start"],  cfg["data_windowing"]["test_runs_end"])

    epochs      = args.epochs or cfg["training"]["epochs"]
    train_bs    = args.train_batch or cfg["training"]["batch"]["train"]
    val_bs      = args.val_batch or cfg["training"]["batch"]["val"]
    test_bs     = args.test_batch or cfg["training"]["batch"]["test"]
    num_workers = args.num_workers or cfg["training"]["num_workers"]

    # ---- Data
    (X_train, y_train, _), (X_test, y_test, _) = load_sampled_data(
        window_size=window_size, stride=stride, ff_path=ff_path, ft_path=ft_path,
        post_fault_start=post_fault_start, train_runs=train_runs, test_runs=test_runs
    )

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    (train_idx, val_idx), = sss.split(X_train, y_train)
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    two_crops = TwoCropsTransform(
        weak_tfms=[TSJitter(0.0005), TSScale(0.99, 1.01)],
        strong_tfms=[TSJitter(0.001), TSScale(0.98, 1.02), TSTimeMask(0.10)]
    )
    train_ds = ContrastiveTSDataset(X_tr, y_tr, two_crops)
    sampler, class_counts = make_sampler(y_tr)

    use_cuda = torch.cuda.is_available()
    use_pin_memory = use_cuda  # only true on GPU

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=train_bs, sampler=sampler, drop_last=True,
        num_workers=num_workers, pin_memory=use_pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        PlainTSDataset(X_val, y_val), batch_size=val_bs, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        PlainTSDataset(X_test, y_test), batch_size=test_bs, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory
    )


    # ---- Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model
    model = SelfGatedHierarchicalTransformerEncoder(
        input_dim=X_train.shape[2], num_classes=int(y_train.max())+1
    ).to(device)

    num_classes = model.classifier[-1].out_features
    with torch.no_grad():
        x0 = next(iter(train_loader))[0].to(device)
        feat_dim = model.forward_features(x0).shape[-1]

    model.cos_head = CosineMarginClassifier(
        feat_dim=feat_dim, num_classes=num_classes,
        s=float(cfg["model"]["s"]), m=float(cfg["model"]["m"]),
        margin_type=str(cfg["model"]["margin_type"])
    ).to(device)

    # per-class margin overrides from config (keys are strings)
    per_m = torch.full((num_classes,), float(cfg["model"]["m"]), device=device)
    for k, v in (cfg.get("model", {}).get("per_class_margin_overrides", {}) or {}).items():
        idx = int(k)
        if idx < num_classes:
            per_m[idx] = float(v)
    model.cos_head.per_class_margin = per_m

    centers = PrototypeCenters(
        num_classes=num_classes, feat_dim=feat_dim,
        momentum=float(cfg["model"]["center_momentum"]), device=device
    ).to(device)

    # per-class center pull weights
    per_class_center_w = torch.ones(num_classes, device=device)
    for k, v in (cfg.get("model", {}).get("center_pull_weights", {}) or {}).items():
        idx = int(k)
        if idx < num_classes:
            per_class_center_w[idx] = float(v)

    center_sep = CenterSeparationLoss(
        K=int(cfg["model"]["center_sep_K"]),
        margin=float(cfg["model"]["center_sep_margin"])
    ).to(device)

    # ---- Optimizer
    opt = torch.optim.AdamW([
        {"params": [p for n,p in model.named_parameters() if not n.startswith("cos_head.")],
         "lr": 3e-4, "weight_decay": 1e-4},
        {"params": [model.cos_head.W], "lr": 3e-4, "weight_decay": 5e-5},
    ], lr=3e-4, weight_decay=0.0)

    # ---- Train
    best_bal_acc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        ce, con, lam = train_one_epoch(
            model, train_loader, opt, device, class_counts,
            base_lambda=0.5, epoch=epoch, total_epochs=epochs,
            mixup_alpha=0.4, mixup_prob=0.35,
            centers=centers, per_class_center_w=per_class_center_w, lambda_center=0.010,
            center_sep=center_sep, lambda_center_sep=0.010, temperature=0.12
        )

        val = evaluate(model, val_loader, device)
        print(f"[{epoch:02d}] λ:{lam:.3f} CE:{ce:.4f} Con:{con:.4f} "
              f"Acc:{val['acc']:.3f} BalAcc:{val['bal_acc']:.3f} F1:{val['macro_f1']:.3f}")

        if val["bal_acc"] > best_bal_acc:
            best_bal_acc = val["bal_acc"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # ---- Test
    if best_state:
        model.load_state_dict(best_state)

    print("=== TEST (best regular) ===")
    test_m = evaluate(model, test_loader, device)
    pretty_print_metrics("TEST", test_m)

    print("=== TEST (best TTA) ===")
    test_tta_m = evaluate_tta(model, test_loader, device, K=8)
    pretty_print_metrics("TEST_TTA", test_tta_m)

    # Save full dicts to file instead of printing
    os.makedirs("results", exist_ok=True)
    with open("results/test_metrics.json", "w") as f:
        json.dump(test_m, f, indent=2, default=lambda x: float(x))
    with open("results/test_metrics_tta.json", "w") as f:
        json.dump(test_tta_m, f, indent=2, default=lambda x: float(x))


if __name__ == "__main__":
    main()
