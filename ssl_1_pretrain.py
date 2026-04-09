"""
Self-supervised pre-training of the ResNet-V2 1D backbone using three
binary pretext tasks: Arrow of Time, Permutation, and Time Warping.

Input data
----------
Reads UNLABELED windows produced by ssl_0_preprocess.py, located at:
    /scratch/bates.car/datasets/paaws_ssl_preprocessed/<location>/unlabeled_windows.npy

The 20 labeled participants are excluded at preprocessing time to avoid
cross-contamination with the downstream fine-tuning set.

Each element is a dict with keys:
    subject_id  (str)
    window_id   (int)
    X, Y, Z     (np.ndarray, shape (300,))  — 30 Hz, 10-second windows
    label       (None — unlabeled)

Checkpoints are saved to:
    /scratch/bates.car/models/ssl_pretrained/<location>/

Training details
----------------
Batch:      4 subjects × 1,500 windows = 6,000 (configurable via CLI)
Optimizer:  Adam, base lr=1e-3, linearly scaled for batch size
LR warmup:  linear burn-in over first 5 epochs
Epochs:     ~20
Split:      8:2 subject-wise train / test
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy import interpolate
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssl_model import ResNet1D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_INPUT_PATH = "/scratch/bates.car/datasets/paaws_ssl_preprocessed/"
BASE_OUTPUT_PATH = "/scratch/bates.car/models/ssl_pretrained/"
LOCATIONS = ["LeftWrist", "RightAnkle", "RightThigh"]

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
FEATURE_DIM = 1024
BASE_LR = 1e-3
REF_BATCH_SIZE = 256        # reference batch size for LR scaling
N_SUBJECTS_PER_BATCH = 4
N_WINDOWS_PER_SUBJECT = 1500
WARMUP_EPOCHS = 5
N_EPOCHS = 20
TRAIN_RATIO = 0.8


# ===========================================================================
# Augmentation (orientation invariance)
# ===========================================================================

def random_axis_swap(signal: np.ndarray) -> np.ndarray:
    """Randomly permute the three accelerometer axes."""
    return signal[np.random.permutation(3), :]


def random_rotation(signal: np.ndarray) -> np.ndarray:
    """Apply a random 3D rotation matrix to the accelerometer axes."""
    a = np.random.uniform(0.0, 2.0 * np.pi, 3)
    cx, cy, cz = np.cos(a)
    sx, sy, sz = np.sin(a)
    Rx = np.array([[1, 0, 0],   [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1,  0],   [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0],[sz, cz, 0],  [0,   0,  1]])
    R = (Rx @ Ry @ Rz).astype(np.float32)
    return R @ signal


def augment(signal: np.ndarray) -> np.ndarray:
    """Random axis swap then random rotation."""
    return random_rotation(random_axis_swap(signal))


# ===========================================================================
# Pretext task transformations
# ===========================================================================

def apply_aot(signal: np.ndarray):
    """Arrow of Time: reverse signal along time axis with 50% probability.

    Returns (transformed_signal, label) where label=1 means reversed.
    """
    if np.random.rand() < 0.5:
        return signal[:, ::-1].copy(), 1
    return signal, 0


def apply_permutation(signal: np.ndarray, n_chunks: int = 4,
                      min_chunk: int = 10):
    """Permutation: split into n_chunks (each ≥ min_chunk samples) and shuffle
    with 50% probability.

    Returns (transformed_signal, label) where label=1 means shuffled.
    """
    if np.random.rand() >= 0.5:
        return signal, 0

    T = signal.shape[1]
    # Distribute T samples into n_chunks with each chunk ≥ min_chunk
    extra = T - n_chunks * min_chunk
    sizes = np.random.multinomial(extra, np.ones(n_chunks) / n_chunks) + min_chunk

    chunks, idx = [], 0
    for size in sizes:
        chunks.append(signal[:, idx: idx + size])
        idx += size

    perm = np.random.permutation(n_chunks)
    while np.all(perm == np.arange(n_chunks)):   # guarantee an actual shuffle
        perm = np.random.permutation(n_chunks)

    return np.concatenate([chunks[i] for i in perm], axis=1), 1


def apply_time_warp(signal: np.ndarray, n_knots: int = 4,
                    sigma: float = 0.2):
    """Time Warping: piecewise-linear stretch/compress via random knot
    perturbation, applied with 50% probability.

    Returns (transformed_signal, label) where label=1 means warped.
    """
    if np.random.rand() >= 0.5:
        return signal, 0

    T = signal.shape[1]
    t = np.arange(T, dtype=float)

    # Interior knot positions (sorted) and their perturbed mappings
    interior = np.sort(np.random.choice(np.arange(1, T - 1),
                                         n_knots, replace=False))
    knot_x = np.concatenate([[0], interior, [T - 1]]).astype(float)
    knot_y = knot_x.copy()
    knot_y[1:-1] += np.random.normal(0.0, sigma * T / n_knots, n_knots)
    knot_y = np.sort(knot_y)        # enforce monotonicity
    knot_y[0] = 0.0
    knot_y[-1] = float(T - 1)

    # Warp function: maps original time indices to new (warped) positions
    warp_fn = interpolate.interp1d(knot_y, knot_x, kind="linear",
                                   bounds_error=False,
                                   fill_value=(knot_x[0], knot_x[-1]))
    warped_t = warp_fn(t)

    warped = np.empty_like(signal)
    for i in range(signal.shape[0]):
        f = interpolate.interp1d(t, signal[i], kind="linear",
                                 bounds_error=False, fill_value="extrapolate")
        warped[i] = f(warped_t).astype(np.float32)

    return warped, 1


# ===========================================================================
# Dataset
# ===========================================================================

class SSLWindowDataset(Dataset):
    """SSL pre-training dataset.

    For each window:
    1. Stack X/Y/Z into (3, 300) and normalize per axis (zero-mean, unit-std).
    2. Apply orientation augmentation (axis swap + rotation).
    3. Apply three independent pretext-task transforms (50/50 each).
    """

    def __init__(self, windows: list):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        signal = np.stack([w["X"], w["Y"], w["Z"]], axis=0).astype(np.float32)

        # Per-axis zero-mean unit-variance normalization
        mu = signal.mean(axis=1, keepdims=True)
        sd = signal.std(axis=1, keepdims=True) + 1e-8
        signal = (signal - mu) / sd

        signal = augment(signal)
        signal, aot_lbl  = apply_aot(signal)
        signal, perm_lbl = apply_permutation(signal)
        signal, tw_lbl   = apply_time_warp(signal)

        return (
            torch.from_numpy(signal),
            torch.tensor(aot_lbl,  dtype=torch.long),
            torch.tensor(perm_lbl, dtype=torch.long),
            torch.tensor(tw_lbl,   dtype=torch.long),
        )


# ===========================================================================
# Weighted batch sampler
# ===========================================================================

def _build_subject_weights(windows: list) -> dict:
    """Return {subject_id: (index_array, weight_array)} with weights
    proportional to per-window signal standard deviation."""
    subject_index: dict = {}
    for i, w in enumerate(windows):
        subject_index.setdefault(w["subject_id"], []).append(i)

    subject_weights = {}
    for sid, indices in subject_index.items():
        weights = np.array([
            np.std(np.stack([windows[i]["X"], windows[i]["Y"], windows[i]["Z"]]))
            for i in indices
        ], dtype=float)
        # Replace NaN/Inf and set a small floor so static windows can still
        # be sampled (but with low probability)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.maximum(weights, 1e-8)
        weights /= weights.sum()
        subject_weights[sid] = (np.array(indices), weights)
    return subject_weights


class SubjectBatchSampler(Sampler):
    """Yields batches by sampling N subjects then M windows per subject
    (with replacement, weighted by signal std dev)."""

    def __init__(self, subject_weights: dict,
                 n_subjects: int = N_SUBJECTS_PER_BATCH,
                 n_windows: int = N_WINDOWS_PER_SUBJECT):
        self.subject_weights = subject_weights
        self.subjects = list(subject_weights.keys())
        self.n_subjects = n_subjects
        self.n_windows = n_windows

    def __len__(self) -> int:
        return len(self.subjects) // self.n_subjects

    def __iter__(self):
        order = np.random.permutation(self.subjects).tolist()
        for b in range(len(self)):
            batch_subjects = order[b * self.n_subjects: (b + 1) * self.n_subjects]
            indices = []
            for sid in batch_subjects:
                idx_arr, w_arr = self.subject_weights[sid]
                sampled = np.random.choice(idx_arr, size=self.n_windows,
                                           replace=True, p=w_arr)
                indices.extend(sampled.tolist())
            yield indices


# ===========================================================================
# Model: backbone + three binary heads
# ===========================================================================

class SSLModel(nn.Module):
    """Shared ResNet backbone with three independent binary softmax heads."""

    def __init__(self, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.backbone  = ResNet1D(feature_dim=feature_dim)
        self.head_aot  = nn.Linear(feature_dim, 2)
        self.head_perm = nn.Linear(feature_dim, 2)
        self.head_tw   = nn.Linear(feature_dim, 2)

    def forward(self, x: torch.Tensor):
        f = self.backbone(x)
        return self.head_aot(f), self.head_perm(f), self.head_tw(f)


# ===========================================================================
# Training helpers
# ===========================================================================

def _run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = {"aot": 0, "perm": 0, "tw": 0}
    total = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for signals, aot_lbl, perm_lbl, tw_lbl in loader:
            signals  = signals.to(device)
            aot_lbl  = aot_lbl.to(device)
            perm_lbl = perm_lbl.to(device)
            tw_lbl   = tw_lbl.to(device)

            aot_out, perm_out, tw_out = model(signals)
            # Equal-weight average of the three cross-entropy losses
            loss = (criterion(aot_out, aot_lbl) +
                    criterion(perm_out, perm_lbl) +
                    criterion(tw_out,  tw_lbl)) / 3.0

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct["aot"]  += (aot_out.argmax(1)  == aot_lbl).sum().item()
            correct["perm"] += (perm_out.argmax(1) == perm_lbl).sum().item()
            correct["tw"]   += (tw_out.argmax(1)   == tw_lbl).sum().item()
            total += len(signals)

    n = max(len(loader), 1)
    acc = {k: v / max(total, 1) for k, v in correct.items()}
    return total_loss / n, acc


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SSL pre-training for a given sensor location.")
    parser.add_argument("--location", required=True, choices=LOCATIONS)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--n-subjects-per-batch", type=int,
                        default=N_SUBJECTS_PER_BATCH)
    parser.add_argument("--n-windows-per-subject", type=int,
                        default=N_WINDOWS_PER_SUBJECT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data loading
    # Input: windows.npy produced by rf_1_preprocess.py
    # ------------------------------------------------------------------
    input_file  = os.path.join(BASE_INPUT_PATH, args.location, "unlabeled_windows.npy")
    output_path = os.path.join(BASE_OUTPUT_PATH, args.location)
    os.makedirs(output_path, exist_ok=True)

    print(f"Loading windows from {input_file} ...")
    windows = np.load(input_file, allow_pickle=True).tolist()
    print(f"  {len(windows):,} windows loaded")

    # 8:2 subject-wise split
    subjects = sorted({w["subject_id"] for w in windows})
    rng = np.random.default_rng(42)
    rng.shuffle(subjects)
    n_train = int(len(subjects) * TRAIN_RATIO)
    train_set = set(subjects[:n_train])
    test_set  = set(subjects[n_train:])

    train_wins = [w for w in windows if w["subject_id"] in train_set]
    test_wins  = [w for w in windows if w["subject_id"] in test_set]
    print(f"  Train: {len(train_set)} subjects, {len(train_wins):,} windows")
    print(f"  Test:  {len(test_set)} subjects, {len(test_wins):,}  windows")

    # Build datasets and loaders
    train_dataset = SSLWindowDataset(train_wins)
    test_dataset  = SSLWindowDataset(test_wins)

    batch_size = args.n_subjects_per_batch * args.n_windows_per_subject
    sw = _build_subject_weights(train_wins)
    batch_sampler = SubjectBatchSampler(sw,
                                        n_subjects=args.n_subjects_per_batch,
                                        n_windows=args.n_windows_per_subject)

    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler,
                              num_workers=min(4, os.cpu_count() or 1), pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=512, shuffle=False,
                              num_workers=min(4, os.cpu_count() or 1), pin_memory=True)

    # ------------------------------------------------------------------
    # Model and optimiser
    # ------------------------------------------------------------------
    model = SSLModel(feature_dim=FEATURE_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Linear LR scaling: lr × (batch_size / ref_batch_size)
    effective_lr = BASE_LR * batch_size / REF_BATCH_SIZE
    optimizer = Adam(model.parameters(), lr=effective_lr)
    print(f"Effective LR: {effective_lr:.5f}  "
          f"(base {BASE_LR} × {batch_size}/{REF_BATCH_SIZE})")

    # Linear warmup: epoch 1 → lr/WARMUP_EPOCHS, epoch 5 → lr
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda ep: min(1.0, (ep + 1) / WARMUP_EPOCHS)
    )

    best_test_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr_loss, _      = _run_epoch(model, train_loader, optimizer, device, train=True)
        te_loss, te_acc = _run_epoch(model, test_loader,  optimizer, device, train=False)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={tr_loss:.4f}  test={te_loss:.4f}  "
              f"aot={te_acc['aot']:.3f}  perm={te_acc['perm']:.3f}  "
              f"tw={te_acc['tw']:.3f}")

        if te_loss < best_test_loss:
            best_test_loss = te_loss
            ckpt = os.path.join(output_path, "best_backbone.pt")
            torch.save({
                "epoch": epoch,
                "backbone_state_dict": model.backbone.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_loss": te_loss,
            }, ckpt)
            print(f"  → saved {ckpt}")

    final_ckpt = os.path.join(output_path, "final_backbone.pt")
    torch.save({"epoch": args.epochs,
                "backbone_state_dict": model.backbone.state_dict()},
               final_ckpt)
    print(f"Final checkpoint saved to {final_ckpt}")


if __name__ == "__main__":
    main()
