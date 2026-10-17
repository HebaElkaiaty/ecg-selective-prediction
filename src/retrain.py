"""Retrain with a fixed final-epoch checkpoint — no validation-based selection.

Addresses R1.1 / R1.2 / R3 Q1 / R4.1: fold 9 is used *only* as a calibration set.
During training, folds 9 and 10 are never loaded, so the nonconformity scores computed
on fold 9 are independent of the model-selection process.

Checkpoint rule (fixed): the final epoch of the one-cycle schedule. OneCycleLR anneals
the learning rate to ~0 by the last step, so the final epoch is the converged one by
design -- this is a principled stopping point, not an arbitrary one.

Fallback (--select-on-fold8): train on folds 1-7 and select the checkpoint on fold 8.
This is triggered ONLY by a failed training curve (diverged / non-decreasing loss).
It must never be triggered by fold-9 or fold-10 results -- doing so would reintroduce
exactly the selection dependence this script exists to remove.

Per-epoch history records training loss and a training-subset AUROC only. No held-out
fold is touched inside the training loop.

Usage
-----
    python -m src.retrain --model acuity35 --seed 0
    python -m src.retrain --model super5   --seed 0

Artifacts land in checkpoints/retrain/<model>_seed<seed>/.
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from config import (
    PTBXL_PATH, HELME_DATA, CHECKPOINT_DIR,
    SAMPLING_RATE, BATCH_SIZE, EPOCHS, LR,
)
from src.utils.data import (
    load_ptbxl_metadata, load_signals, get_fold_splits,
    aggregate_labels, apply_standardizer, load_standardizer,
)
from src.models.resnet1d import ECGResNet

ACUITY_CLASSES = 35
SUPER_CLASSES  = 5
CAL_FOLD       = 9    # calibration fold -- never loaded during training
TEST_FOLD      = 10   # test fold        -- never loaded during training


class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def safe_macro_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Macro AUROC, skipping classes with a single label value."""
    per_class = [
        roc_auc_score(labels[:, c], probs[:, c])
        for c in range(labels.shape[1])
        if len(np.unique(labels[:, c])) >= 2
    ]
    return float(np.mean(per_class)) if per_class else float("nan")


@torch.no_grad()
def predict_logits(model, X: np.ndarray, device, batch_size: int = BATCH_SIZE) -> np.ndarray:
    model.eval()
    dl = DataLoader(TensorDataset(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))),
                    batch_size=batch_size, shuffle=False, num_workers=0)
    out = [model(b.to(device)).cpu().numpy() for (b,) in dl]
    return np.concatenate(out, axis=0)


@torch.no_grad()
def predict_mc_probs(model, X: np.ndarray, device, n_passes: int,
                     n_classes: int, batch_size: int = BATCH_SIZE) -> np.ndarray:
    """MC-dropout mean sigmoid probabilities.

    BatchNorm stays in eval mode (running statistics); only Dropout goes stochastic.
    Mirrors src/calibration/mc_dropout.py exactly.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    dl = DataLoader(TensorDataset(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))),
                    batch_size=batch_size, shuffle=False, num_workers=0)
    acc = np.zeros((n_passes, X.shape[0], n_classes), dtype=np.float32)
    for p in range(n_passes):
        offset = 0
        for (b,) in dl:
            logits = model(b.to(device)).cpu().numpy()
            n = logits.shape[0]
            acc[p, offset:offset + n] = 1.0 / (1.0 + np.exp(-logits))
            offset += n
        if (p + 1) % 10 == 0:
            print(f"      MC pass {p + 1}/{n_passes}")
    model.eval()
    return acc.mean(axis=0)


def build_labels(model_name: str, meta: pd.DataFrame, splits: dict):
    """Return (Y_full, class_names) with Y_full aligned row-for-row with meta."""
    n = len(meta)
    if model_name == "super5":
        scp = pd.read_csv(Path(PTBXL_PATH) / "scp_statements.csv", index_col=0)
        labelled = aggregate_labels(meta, scp, task="super")
        cols = [c for c in labelled.columns
                if c not in list(meta.columns) + ["label_primary"]]
        Y = labelled[cols].values.astype(np.float32)
        assert Y.shape == (n, SUPER_CLASSES), f"super5 labels {Y.shape}"
        return Y, list(cols)

    # acuity35 -- stored pre-split; scatter back to full length
    lab_dir = Path(CHECKPOINT_DIR) / "acuity_labels"
    with open(lab_dir / "class_names.json") as f:
        class_names = json.load(f)
    assert len(class_names) == ACUITY_CLASSES
    Y = np.zeros((n, ACUITY_CLASSES), dtype=np.float32)
    for key, fname in [("train", "y_combined_train.npy"),
                       ("val",   "y_combined_val.npy"),
                       ("test",  "y_combined_test.npy")]:
        part = np.load(lab_dir / fname)
        assert part.shape[0] == len(splits[key]), f"{key}: {part.shape[0]} vs {len(splits[key])}"
        Y[splits[key]] = part
    return Y, class_names


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["acuity35", "super5"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--mc-passes", type=int, default=30,
                    help="MC-dropout passes (default 30, matching the original run)")
    ap.add_argument("--train-auc-n", type=int, default=2000,
                    help="training records sampled for the per-epoch diagnostic AUROC")
    ap.add_argument("--select-on-fold8", action="store_true",
                    help="FALLBACK ONLY: train folds 1-7, select checkpoint on fold 8. "
                         "Trigger on a failed training curve, never on fold-9/10 results.")
    ap.add_argument("--out-root", default=str(Path(CHECKPOINT_DIR) / "retrain"))
    ap.add_argument("--dry-run", action="store_true",
                    help="validate data, splits and one forward pass, then exit "
                         "without training or writing artifacts")
    args = ap.parse_args()

    n_classes = ACUITY_CLASSES if args.model == "acuity35" else SUPER_CLASSES
    # checkpoint rule is part of the run identity -- the two modes must never
    # overwrite each other's artifacts
    mode = "fold8" if args.select_on_fold8 else "final"
    out_dir = Path(args.out_root) / f"{args.model}_seed{args.seed}_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Model: {args.model} ({n_classes} classes) | seed {args.seed} | device {device}")
    print(f"Output: {out_dir}")

    # ---- data ---------------------------------------------------------------
    meta = load_ptbxl_metadata(PTBXL_PATH)
    splits = get_fold_splits(meta)

    cache = Path(CHECKPOINT_DIR) / "X_raw.npy"
    if cache.exists():
        print(f"Loading cached signals from {cache}")
        X = np.load(cache)
    else:
        print("Loading signals from disk (one-time)...")
        X = load_signals(meta, PTBXL_PATH, sampling_rate=SAMPLING_RATE)
        np.save(cache, X)

    Y, class_names = build_labels(args.model, meta, splits)

    scaler = load_standardizer(Path(HELME_DATA) / "standard_scaler.pkl")
    X = apply_standardizer(X, scaler)
    X = np.ascontiguousarray(X.transpose(0, 2, 1))   # (N, 12, T)

    folds = meta["strat_fold"].values
    if args.select_on_fold8:
        train_idx = np.where(folds <= 7)[0]
        sel_idx   = np.where(folds == 8)[0]
    else:
        train_idx = np.where(folds <= 8)[0]
        sel_idx   = None
    cal_idx  = np.where(folds == CAL_FOLD)[0]
    test_idx = np.where(folds == TEST_FOLD)[0]

    assert not set(train_idx) & set(cal_idx), "train/calibration overlap"
    assert not set(train_idx) & set(test_idx), "train/test overlap"
    np.testing.assert_array_equal(cal_idx, splits["val"])
    np.testing.assert_array_equal(test_idx, splits["test"])
    print(f"train={len(train_idx)}  "
          f"select={0 if sel_idx is None else len(sel_idx)}  "
          f"cal(fold9)={len(cal_idx)}  test(fold10)={len(test_idx)}")

    X_train, Y_train = X[train_idx], Y[train_idx]

    # Fixed training subset for the per-epoch diagnostic AUROC. Training data only.
    rng = np.random.default_rng(args.seed)
    sub = rng.choice(len(train_idx), size=min(args.train_auc_n, len(train_idx)),
                     replace=False)
    X_sub, Y_sub = X_train[sub], Y_train[sub]

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(ECGDataset(X_train, Y_train), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, generator=g)

    # ---- train --------------------------------------------------------------
    model = ECGResNet(num_classes=n_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=LR,
                           steps_per_epoch=len(train_loader), epochs=args.epochs)

    if args.dry_run:
        xb = torch.from_numpy(np.ascontiguousarray(X_train[:BATCH_SIZE], dtype=np.float32))
        with torch.no_grad():
            out = model(xb.to(device))
        print(f"DRY RUN: forward {tuple(xb.shape)} -> {tuple(out.shape)}, "
              f"{len(train_loader)} steps/epoch x {args.epochs} epochs")
        print("DRY RUN: data, splits and model OK. No training, no artifacts written.")
        return

    history, best_sel, t0 = [], -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{args.epochs}",
                   leave=False, unit="batch")
        for Xb, yb in bar:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total / len(train_loader)
        sub_auc = safe_macro_auc(Y_sub, 1 / (1 + np.exp(-predict_logits(model, X_sub, device))))
        row = {"epoch": epoch, "train_loss": round(train_loss, 6),
               "train_subset_auroc": round(sub_auc, 6),
               "lr": scheduler.get_last_lr()[0]}

        if args.select_on_fold8:
            sel_auc = safe_macro_auc(
                Y[sel_idx], 1 / (1 + np.exp(-predict_logits(model, X[sel_idx], device))))
            row["fold8_auroc"] = round(sel_auc, 6)
            if sel_auc > best_sel:
                best_sel = sel_auc
                torch.save(model.state_dict(), out_dir / "model_selected.pt")

        history.append(row)
        print(f"Epoch {epoch:2d}/{args.epochs} | loss {train_loss:.4f} | "
              f"train-subset AUROC {sub_auc:.4f}"
              + (f" | fold8 AUROC {row['fold8_auroc']:.4f}" if args.select_on_fold8 else ""))

    train_secs = time.time() - t0
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    # Checkpoint rule
    torch.save(model.state_dict(), out_dir / "model_final.pt")
    if args.select_on_fold8:
        model.load_state_dict(torch.load(out_dir / "model_selected.pt", weights_only=True))
        ckpt_rule = "fold8_selected (FALLBACK)"
    else:
        ckpt_rule = "final_epoch"
    print(f"\nCheckpoint rule: {ckpt_rule} | training {train_secs/60:.1f} min")

    # ---- fold 9 / fold 10 are touched only from here on ---------------------
    (out_dir / "class_order.txt").write_text("\n".join(class_names) + "\n")

    summary = {}
    for tag, idx in [("fold9", cal_idx), ("fold10", test_idx)]:
        logits = predict_logits(model, X[idx], device)
        scores = 1.0 / (1.0 + np.exp(-logits))
        labels = Y[idx]
        np.save(out_dir / f"logits_{tag}.npy", logits)
        np.save(out_dir / f"scores_{tag}.npy", scores)
        np.save(out_dir / f"labels_{tag}.npy", labels)

        print(f"  {tag}: running {args.mc_passes} MC-dropout passes ...")
        np.save(out_dir / f"mc_probs_{tag}.npy",
                predict_mc_probs(model, X[idx], device, args.mc_passes, n_classes))

        df = meta.iloc[idx][["ecg_id", "patient_id"]].reset_index(drop=True)
        for j, c in enumerate(class_names):
            df[c] = labels[:, j].astype(int)
        df.to_csv(out_dir / f"meta_{tag}.csv", index=False)

        summary[f"{tag}_macro_auroc"] = round(safe_macro_auc(labels, scores), 6)
        print(f"  {tag}: macro AUROC {summary[f'{tag}_macro_auroc']:.4f}  "
              f"logits {logits.shape}")

    cfg = {
        "model": args.model, "n_classes": n_classes, "seed": args.seed,
        "epochs": args.epochs, "checkpoint_rule": ckpt_rule,
        "select_on_fold8": args.select_on_fold8,
        "train_folds": "1-7" if args.select_on_fold8 else "1-8",
        "n_train": int(len(train_idx)), "n_cal": int(len(cal_idx)),
        "n_test": int(len(test_idx)),
        "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": 1e-4,
        "scheduler": "OneCycleLR", "optimizer": "AdamW",
        "mc_passes": args.mc_passes, "device": str(device),
        "train_minutes": round(train_secs / 60, 2),
        "final_train_loss": history[-1]["train_loss"],
        "final_train_subset_auroc": history[-1]["train_subset_auroc"],
        **summary,
    }
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"\nSaved to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
