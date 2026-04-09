"""
Downstream fine-tuning of the pre-trained ResNet-V2 backbone.

Input data
----------
Labeled windows (the 20 labeled participants, separate from SSL pre-training),
produced by ssl_0_preprocess.py at:
    /scratch/bates.car/datasets/paaws_ssl_preprocessed/<location>/labeled_windows.npy

Pre-trained backbone checkpoint, read from:
    /scratch/bates.car/models/ssl_pretrained/<location>/best_backbone.pt
    (override with --checkpoint)

Results are saved to:
    /scratch/bates.car/datasets/paaws_ssl_results/<location>/

Architecture
------------
Pre-trained ResNet backbone → FC(1024, 512) → ReLU → FC(512, n_classes)
All layers are fine-tuned (no frozen trunk).

Validation protocol
-------------------
≥10 subjects : 5-fold subject-wise CV, 7:1:2 train/val/test
< 10 subjects: leave-one-subject-out
Early stopping with patience=5 on validation loss.

Metrics: subject-wise macro F1, pooled macro F1, Cohen's Kappa (κ)
"""

import argparse
import os
import sys

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssl_model import ResNet1D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_INPUT_PATH   = "/scratch/bates.car/datasets/paaws_ssl_preprocessed/"
BASE_PRETRAIN_PATH = "/scratch/bates.car/models/ssl_pretrained/"
BASE_OUTPUT_PATH  = "/scratch/bates.car/datasets/paaws_ssl_results/"
LOCATIONS = ["LeftWrist", "RightAnkle", "RightThigh"]

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
FEATURE_DIM = 1024
FC_DIM      = 512
LR          = 1e-3
BATCH_SIZE  = 256
N_FOLDS     = 5
PATIENCE    = 5
MAX_EPOCHS  = 200


# ===========================================================================
# Dataset
# ===========================================================================

class LabeledWindowDataset(Dataset):
    """Labeled accelerometer windows. Normalizes per axis (zero-mean, unit-std)."""

    def __init__(self, windows: list, label_encoder: LabelEncoder):
        self.windows     = windows
        self.le          = label_encoder
        self.subject_ids = [w["subject_id"] for w in windows]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        w      = self.windows[idx]
        signal = np.stack([w["X"], w["Y"], w["Z"]], axis=0).astype(np.float32)
        mu     = signal.mean(axis=1, keepdims=True)
        sd     = signal.std(axis=1, keepdims=True) + 1e-8
        signal = (signal - mu) / sd
        label  = int(self.le.transform([w["label"]])[0])
        return torch.from_numpy(signal), torch.tensor(label, dtype=torch.long)


# ===========================================================================
# Model
# ===========================================================================

class FineTuneModel(nn.Module):
    """Pre-trained backbone + two-layer classification head."""

    def __init__(self, backbone: ResNet1D, n_classes: int,
                 feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.backbone   = backbone
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, FC_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(FC_DIM, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


# ===========================================================================
# Cross-validation splits
# ===========================================================================

def subject_wise_cv_splits(subjects: list, n_folds: int):
    """Yield (train_subjects, val_subjects, test_subjects) for each fold.

    Split ratio 7:1:2. The test window rotates across folds so every subject
    appears in the test set exactly once (approximately).
    """
    n       = len(subjects)
    n_test  = max(1, round(n * 0.2))
    n_val   = max(1, round(n * 0.1))

    step = max(1, n // n_folds)
    for fold in range(n_folds):
        rotated = subjects[fold * step:] + subjects[:fold * step]
        test    = rotated[:n_test]
        val     = rotated[n_test: n_test + n_val]
        train   = rotated[n_test + n_val:]
        yield train, val, test


def loso_splits(subjects: list):
    """Leave-one-subject-out splits."""
    for i, s in enumerate(subjects):
        yield [x for j, x in enumerate(subjects) if j != i], [], [s]


# ===========================================================================
# Training with early stopping
# ===========================================================================

def train_fold(model, train_loader, val_loader, device) -> nn.Module:
    criterion       = nn.CrossEntropyLoss()
    optimizer       = Adam(model.parameters(), lr=LR)
    best_val_loss   = float("inf")
    best_state      = None
    patience_counter = 0

    for _ in range(MAX_EPOCHS):
        model.train()
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(signals), labels).backward()
            optimizer.step()

        # Validation
        if len(val_loader) == 0:
            break
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                val_loss += criterion(model(signals), labels).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ===========================================================================
# Per-fold evaluation
# ===========================================================================

def run_fold(fold_num: int,
             train_wins: list, val_wins: list, test_wins: list,
             le: LabelEncoder, backbone_state: dict,
             device: torch.device) -> dict:

    n_classes = len(le.classes_)

    def make_loader(wins, shuffle):
        return DataLoader(LabeledWindowDataset(wins, le),
                          batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=4, pin_memory=True)

    train_loader = make_loader(train_wins, shuffle=True)
    val_loader   = make_loader(val_wins,   shuffle=False) if val_wins else []
    test_loader  = make_loader(test_wins,  shuffle=False)

    backbone = ResNet1D(feature_dim=FEATURE_DIM)
    backbone.load_state_dict(backbone_state)
    model = FineTuneModel(backbone, n_classes=n_classes).to(device)

    model = train_fold(model, train_loader, val_loader, device)

    # Collect test predictions
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            preds   = model(signals).argmax(1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    # Subject IDs aligned with test_wins (loader is unshuffled)
    subj_ids = [w["subject_id"] for w in test_wins]

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa    = cohen_kappa_score(y_true, y_pred)
    sw_f1    = _subject_wise_f1(y_true, y_pred, subj_ids)

    print(f"  Fold {fold_num}: macro_F1={macro_f1:.4f}  "
          f"subject_F1={sw_f1:.4f}  kappa={kappa:.4f}")
    return {"fold": fold_num, "macro_f1": macro_f1,
            "subject_f1": sw_f1, "kappa": kappa}


def _subject_wise_f1(y_true: list, y_pred: list, subj_ids: list) -> float:
    """Compute macro F1 per subject then average (subject-wise F1)."""
    subjects = sorted(set(subj_ids))
    f1s = []
    for sid in subjects:
        mask = [i for i, s in enumerate(subj_ids) if s == sid]
        yt   = [y_true[i] for i in mask]
        yp   = [y_pred[i] for i in mask]
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
    return float(np.mean(f1s))


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SSL fine-tuning for a given sensor location.")
    parser.add_argument("--location", required=True, choices=LOCATIONS)
    parser.add_argument("--checkpoint", default="best_backbone.pt",
                        help="Checkpoint filename inside BASE_PRETRAIN_PATH/<location>/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    input_file    = os.path.join(BASE_INPUT_PATH,    args.location, "labeled_windows.npy")
    pretrain_ckpt = os.path.join(BASE_PRETRAIN_PATH, args.location, args.checkpoint)
    output_path   = os.path.join(BASE_OUTPUT_PATH,   args.location)
    os.makedirs(output_path, exist_ok=True)

    print(f"Loading windows from {input_file} ...")
    windows = np.load(input_file, allow_pickle=True).tolist()
    windows = [w for w in windows if w.get("label") is not None]
    print(f"  {len(windows):,} labeled windows")

    print(f"Loading backbone from {pretrain_ckpt} ...")
    ckpt           = torch.load(pretrain_ckpt, map_location="cpu")
    backbone_state = ckpt["backbone_state_dict"]

    # Encode labels
    le = LabelEncoder()
    le.fit([w["label"] for w in windows])
    print(f"Classes ({len(le.classes_)}): {le.classes_.tolist()}")

    subjects = sorted({w["subject_id"] for w in windows})
    print(f"Subjects: {len(subjects)}")

    # Build subject → window list index
    subj_wins: dict = {}
    for w in windows:
        subj_wins.setdefault(w["subject_id"], []).append(w)

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------
    if len(subjects) >= 10:
        print(f"5-fold subject-wise CV (7:1:2 split)")
        splits = list(subject_wise_cv_splits(subjects, N_FOLDS))
    else:
        print(f"Leave-one-subject-out CV ({len(subjects)} subjects < 10)")
        splits = list(loso_splits(subjects))

    all_results = []
    for fold_num, (tr_s, val_s, te_s) in enumerate(splits, start=1):
        print(f"\n--- Fold {fold_num}/{len(splits)} ---")
        print(f"  Train: {len(tr_s)}, Val: {len(val_s)}, Test: {len(te_s)}")

        tr_wins  = [w for s in tr_s  for w in subj_wins[s]]
        val_wins = [w for s in val_s for w in subj_wins[s]]
        te_wins  = [w for s in te_s  for w in subj_wins[s]]

        result = run_fold(fold_num, tr_wins, val_wins, te_wins,
                          le, backbone_state, device)
        all_results.append(result)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    mf1s  = [r["macro_f1"]   for r in all_results]
    swf1s = [r["subject_f1"] for r in all_results]
    kaps  = [r["kappa"]      for r in all_results]
    print(f"Macro F1 (pooled):  {np.mean(mf1s):.4f} ± {np.std(mf1s):.4f}")
    print(f"Subject-wise F1:    {np.mean(swf1s):.4f} ± {np.std(swf1s):.4f}")
    print(f"Cohen's Kappa (κ):  {np.mean(kaps):.4f} ± {np.std(kaps):.4f}")

    out_csv = os.path.join(output_path, "cv_results.csv")
    pl.DataFrame(all_results).write_csv(out_csv)
    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    main()
