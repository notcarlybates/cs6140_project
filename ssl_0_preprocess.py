"""
SSL data preprocessing.

Two separate outputs:
  1. unlabeled_windows.npy  — unlabeled participants for SSL pre-training
  2. labeled_windows.npy    — the 20 labeled participants for downstream fine-tuning

The 20 labeled participants (identified by their presence in the RF pipeline's
trimmed directory) are EXCLUDED from the unlabeled set to prevent
cross-contamination.

Unlabeled data source
---------------------
Raw ActiGraph CSVs at:
    /scratch/mazzucchelli.u/paaws_fl_data/acc_data/DS_<id>/
        OriginalRaw/ActiGraph/csv_synced/DS_<id>-Free-<Location><date>.csv

Labeled data source
-------------------
Already-synced CSVs produced by the RF pipeline at:
    /scratch/bates.car/datasets/paaws_fl_trimmed/<Location>/<id>_synced.csv

Output directory
----------------
    /scratch/bates.car/datasets/paaws_ssl_preprocessed/<Location>/
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import polars as pl
from scipy import interpolate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_accelerometer_data import parse_header
from utils import MAPPING_SCHEMES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
UNLABELED_ROOT = "/scratch/mazzucchelli.u/paaws_fl_data/acc_data/"
LABELED_ROOT   = "/scratch/bates.car/datasets/paaws_fl_trimmed/"
OUTPUT_ROOT    = "/scratch/bates.car/datasets/paaws_ssl_preprocessed/"

LOCATIONS = ["LeftWrist", "RightAnkle", "RightThigh"]

TARGET_HZ   = 30
WINDOW_SEC  = 10
WINDOW_SIZE = TARGET_HZ * WINDOW_SEC   # 300 samples

ACTIVITY_MAPPING = MAPPING_SCHEMES["lab_fl_5"]


# ===========================================================================
# Shared helpers (adapted from rf_1_preprocess.py)
# ===========================================================================

def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """Fill NaN values in a 1-D array via linear interpolation.

    Leading/trailing NaNs are forward/back-filled with the nearest valid value.
    If the entire array is NaN it is returned unchanged.
    """
    nans = np.isnan(arr)
    if not nans.any():
        return arr
    if nans.all():
        return arr
    valid = ~nans
    arr = arr.copy()
    arr[nans] = np.interp(
        np.flatnonzero(nans), np.flatnonzero(valid), arr[valid]
    )
    return arr


def resample_to_30hz(data: np.ndarray, original_hz: int) -> np.ndarray:
    """Linearly resample a (3, N) array from original_hz to 30 Hz.

    NaN values in the input are interpolated over before resampling so they
    do not propagate.
    """
    # Interpolate over NaN gaps per axis
    for ax in range(data.shape[0]):
        data[ax] = _interpolate_nans(data[ax])

    if original_hz == TARGET_HZ:
        return data
    n_orig = data.shape[1]
    n_target = int(n_orig * TARGET_HZ / original_hz)
    orig_idx = np.linspace(0, n_orig - 1, n_orig)
    tgt_idx  = np.linspace(0, n_orig - 1, n_target)
    out = np.empty((3, n_target), dtype=np.float32)
    for ax in range(3):
        f = interpolate.interp1d(orig_idx, data[ax], kind="linear")
        out[ax] = f(tgt_idx)
    return out


def make_windows(data: np.ndarray, subject_id: str,
                 labels: np.ndarray | None = None) -> list[dict]:
    """Split a (3, T) array into non-overlapping 10-second windows.

    If *labels* is provided (length T), each window gets a majority-vote label.
    Otherwise the label is set to None (unlabeled data).
    """
    T = data.shape[1]
    n_windows = T // WINDOW_SIZE
    windows = []

    for i in range(n_windows):
        s = i * WINDOW_SIZE
        e = s + WINDOW_SIZE
        x_seg, y_seg, z_seg = data[0, s:e], data[1, s:e], data[2, s:e]

        # Drop windows that still contain NaN (e.g. entirely-NaN axis)
        if np.isnan(x_seg).any() or np.isnan(y_seg).any() or np.isnan(z_seg).any():
            continue

        w = {
            "subject_id": subject_id,
            "window_id":  i,
            "X": x_seg,
            "Y": y_seg,
            "Z": z_seg,
        }
        if labels is not None:
            seg = labels[s:e]
            # Majority-vote label (skip None/null)
            valid = [v for v in seg if v is not None]
            if not valid:
                continue
            w["label"] = max(set(valid), key=valid.count)
        else:
            w["label"] = None
        windows.append(w)

    return windows


def _accel_columns(df: pl.DataFrame) -> list[str]:
    """Return the three accelerometer column names in X, Y, Z order."""
    candidates = [
        ("Accelerometer X", "Accelerometer Y", "Accelerometer Z"),
        ("X", "Y", "Z"),
    ]
    for cx, cy, cz in candidates:
        if cx in df.columns and cy in df.columns and cz in df.columns:
            return [cx, cy, cz]
    raise ValueError(f"Cannot find accel columns in {df.columns}")


# ===========================================================================
# Discover participant IDs
# ===========================================================================

def get_labeled_ids(location: str) -> set[str]:
    """Return the set of participant IDs that have labeled data."""
    trimmed_dir = os.path.join(LABELED_ROOT, location)
    if not os.path.isdir(trimmed_dir):
        return set()
    ids = set()
    for f in os.listdir(trimmed_dir):
        if f.endswith("_synced.csv"):
            # e.g. "DS_1001_synced.csv" → "DS_1001"
            ids.add(f.replace("_synced.csv", ""))
    return ids


def discover_unlabeled_files(location: str, exclude: set[str]) -> list[tuple[str, str]]:
    """Return [(filepath, subject_id), ...] for unlabeled participants.

    File pattern:
        <UNLABELED_ROOT>/DS_<id>/OriginalRaw/ActiGraph/csv_synced/
            DS_<id>-Free-<Location><date>.csv

    Any subject_id in *exclude* is skipped.
    """
    pattern = os.path.join(
        UNLABELED_ROOT,
        "DS_*",
        "OriginalRaw", "ActiGraph", "csv_synced",
        f"DS_*-Free-{location}*.csv",
    )
    results = []
    for path in sorted(glob.glob(pattern)):
        basename = os.path.basename(path)
        # Extract subject ID (DS_XXXX) from the filename
        m = re.match(r"(DS_\d+)-Free-", basename)
        if m is None:
            continue
        sid = m.group(1)
        if sid in exclude:
            continue
        results.append((path, sid))
    return results


# ===========================================================================
# Preprocess one participant
# ===========================================================================

def preprocess_unlabeled(filepath: str, subject_id: str) -> list[dict]:
    """Read a raw ActiGraph CSV (no labels), resample, and window."""
    print(f"  [unlabeled] {subject_id}  ({filepath})")
    try:
        _, sampling_rate = parse_header(filepath)
    except Exception:
        sampling_rate = 80      # fallback
    df = pl.read_csv(filepath, skip_rows=10)
    cols = _accel_columns(df)
    data = np.stack([df[c].to_numpy().astype(np.float32) for c in cols])
    data = resample_to_30hz(data, sampling_rate)
    return make_windows(data, subject_id, labels=None)


def preprocess_labeled(filepath: str, subject_id: str) -> list[dict]:
    """Read a synced-and-trimmed CSV (with Activity column), resample, and window."""
    print(f"  [labeled]   {subject_id}  ({filepath})")
    df = pl.read_csv(filepath)

    # Apply activity mapping; drop unmapped / null rows
    if "Activity" in df.columns:
        df = df.filter(
            pl.col("Activity").is_not_null()
            & ~pl.col("Activity").is_in(
                ["Unknown", "Before_Data_Collection", "After_Data_Collection"]
            )
        ).with_columns(
            pl.col("Activity")
            .replace(ACTIVITY_MAPPING, default=None)
            .alias("Activity")
        ).filter(pl.col("Activity").is_not_null())

    cols = _accel_columns(df)
    data = np.stack([df[c].to_numpy().astype(np.float32) for c in cols])
    labels = df["Activity"].to_list() if "Activity" in df.columns else None

    # Labeled data is 80 Hz (same as unlabeled)
    data = resample_to_30hz(data, original_hz=80)

    # Resample labels to match 30 Hz length
    if labels is not None:
        n_orig = len(labels)
        n_target = data.shape[1]
        idx_map = np.round(
            np.linspace(0, n_orig - 1, n_target)
        ).astype(int)
        labels = [labels[i] for i in idx_map]

    return make_windows(data, subject_id, labels=labels)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SSL preprocessing: separate unlabeled and labeled windows.")
    parser.add_argument("--location", required=True, choices=LOCATIONS,
                        help="Sensor location to process")
    args = parser.parse_args()
    location = args.location

    out_dir = os.path.join(OUTPUT_ROOT, location)
    os.makedirs(out_dir, exist_ok=True)

    # ----- discover participants -----
    labeled_ids = get_labeled_ids(location)
    print(f"Labeled participants ({len(labeled_ids)}): {sorted(labeled_ids)}")

    unlabeled_files = discover_unlabeled_files(location, exclude=labeled_ids)
    print(f"Unlabeled participants (after exclusion): {len(unlabeled_files)}")

    # ----- preprocess unlabeled (for SSL pre-training) -----
    print(f"\n--- Unlabeled data ({location}) ---")
    unlabeled_windows: list[dict] = []
    for filepath, sid in unlabeled_files:
        try:
            wins = preprocess_unlabeled(filepath, sid)
            unlabeled_windows.extend(wins)
        except Exception as e:
            print(f"    ERROR {sid}: {e}")

    ul_path = os.path.join(out_dir, "unlabeled_windows.npy")
    np.save(ul_path, unlabeled_windows, allow_pickle=True)
    n_ul_subj = len({w["subject_id"] for w in unlabeled_windows})
    print(f"Saved {len(unlabeled_windows):,} unlabeled windows "
          f"({n_ul_subj} subjects) → {ul_path}")

    # ----- preprocess labeled (for downstream fine-tuning) -----
    print(f"\n--- Labeled data ({location}) ---")
    labeled_dir = os.path.join(LABELED_ROOT, location)
    labeled_windows: list[dict] = []
    for f in sorted(os.listdir(labeled_dir)):
        if not f.endswith("_synced.csv"):
            continue
        sid = f.replace("_synced.csv", "")
        fp  = os.path.join(labeled_dir, f)
        try:
            wins = preprocess_labeled(fp, sid)
            labeled_windows.extend(wins)
        except Exception as e:
            print(f"    ERROR {sid}: {e}")

    lb_path = os.path.join(out_dir, "labeled_windows.npy")
    np.save(lb_path, labeled_windows, allow_pickle=True)
    n_lb_subj = len({w["subject_id"] for w in labeled_windows})
    print(f"Saved {len(labeled_windows):,} labeled windows "
          f"({n_lb_subj} subjects) → {lb_path}")


if __name__ == "__main__":
    main()
