"""
Preprocessing script for HAR Random Forest pipeline.
1. Resample accelerometer data to 30Hz
2. Window into 10-second segments
"""

import argparse
import os
import sys
import numpy as np
import polars as pl
from scipy import interpolate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import MAPPING_SCHEMES

BASE_INPUT_PATH = "/scratch/bates.car/datasets/paaws_fl_trimmed/"
BASE_OUTPUT_PATH = "/scratch/bates.car/datasets/paaws_fl_preprocessed/"
LOCATIONS = ["LeftWrist", "RightAnkle", "RightThigh"]
TARGET_HZ = 30
WINDOW_SEC = 10
WINDOW_SIZE = TARGET_HZ * WINDOW_SEC  # 300 samples per window

ACTIVITY_MAPPING = MAPPING_SCHEMES["lab_fl_5"]


def resample_to_30hz(df: pl.DataFrame, original_hz: int = 30) -> pl.DataFrame:
    """Linearly resample accelerometer data to 30Hz."""
    if original_hz == TARGET_HZ:
        return df

    # Get acceleration columns (assuming X, Y, Z naming)
    accel_cols = [c for c in df.columns if c in ["X", "Y", "Z", "Accelerometer X", "Accelerometer Y", "Accelerometer Z"]]

    n_original = len(df)
    n_target = int(n_original * TARGET_HZ / original_hz)

    original_indices = np.linspace(0, n_original - 1, n_original)
    target_indices = np.linspace(0, n_original - 1, n_target)

    resampled_data = {}
    for col in accel_cols:
        f = interpolate.interp1d(original_indices, df[col].to_numpy(), kind='linear')
        resampled_data[col] = f(target_indices)

    # Handle Activity column (take nearest)
    if "Activity" in df.columns:
        activity_indices = np.round(target_indices).astype(int)
        activity_indices = np.clip(activity_indices, 0, n_original - 1)
        resampled_data["Activity"] = df["Activity"].to_numpy()[activity_indices]

    return pl.DataFrame(resampled_data)


def create_windows(df: pl.DataFrame, subject_id: str) -> list[dict]:
    """Split data into 10-second windows."""
    accel_cols = [c for c in df.columns if c in ["X", "Y", "Z", "Accelerometer X", "Accelerometer Y", "Accelerometer Z"]]

    # Standardize column names
    col_map = {}
    for c in accel_cols:
        if "X" in c:
            col_map[c] = "X"
        elif "Y" in c:
            col_map[c] = "Y"
        elif "Z" in c:
            col_map[c] = "Z"
    df = df.rename(col_map)

    windows = []
    n_samples = len(df)
    n_windows = n_samples // WINDOW_SIZE

    for i in range(n_windows):
        start = i * WINDOW_SIZE
        end = start + WINDOW_SIZE

        window_df = df.slice(start, WINDOW_SIZE)

        # Get majority activity label for this window
        if "Activity" in window_df.columns:
            activity_counts = window_df.group_by("Activity").len().sort("len", descending=True)
            label = activity_counts["Activity"][0]

            # Skip windows with null labels (should not occur after upstream filtering)
            if label is None:
                continue
        else:
            label = None

        windows.append({
            "subject_id": subject_id,
            "window_id": i,
            "X": window_df["X"].to_numpy(),
            "Y": window_df["Y"].to_numpy(),
            "Z": window_df["Z"].to_numpy(),
            "label": label
        })

    return windows


def preprocess_subject(filepath: str, subject_id: str) -> list[dict]:
    """Load, resample, and window a single subject's data."""
    print(f"  Loading {subject_id}...")
    df = pl.read_csv(filepath)

    # Drop rows with invalid/unknown activity labels before resampling and windowing,
    # then apply the 5-activity mapping (drop rows whose raw label isn't in the mapping).
    if "Activity" in df.columns:
        before = len(df)
        df = df.filter(
            pl.col("Activity").is_not_null() &
            ~pl.col("Activity").is_in(["Unknown", "Before_Data_Collection", "After_Data_Collection"])
        )
        # Map raw activity labels to the 5 canonical activities; drop unmapped rows
        df = df.with_columns(
            pl.col("Activity").replace(ACTIVITY_MAPPING, default=None).alias("Activity")
        ).filter(pl.col("Activity").is_not_null())
        print(f"  Dropped {before - len(df):,} rows with invalid/unmapped labels ({len(df):,} remaining)")

    print(f"  Resampling from 80Hz to {TARGET_HZ}Hz...")
    df = resample_to_30hz(df, original_hz=80)

    print(f"  Creating {WINDOW_SEC}s windows...")
    windows = create_windows(df, subject_id)

    print(f"  Created {len(windows)} windows")
    return windows


def main():
    parser = argparse.ArgumentParser(description="Preprocess HAR data for a given sensor location.")
    parser.add_argument("--location", required=True, choices=LOCATIONS,
                        help="Sensor location to process (LeftWrist, RightAnkle, RightThigh)")
    args = parser.parse_args()
    location = args.location

    input_path = os.path.join(BASE_INPUT_PATH, location) + "/"
    output_path = os.path.join(BASE_OUTPUT_PATH, location) + "/"

    print(f"Location:    {location}")
    print(f"Input path:  {input_path}")
    print(f"Output path: {output_path}")

    os.makedirs(output_path, exist_ok=True)

    all_windows = []

    files = sorted([f for f in os.listdir(input_path) if f.endswith(".csv")])
    print(f"Found {len(files)} subjects")

    for file in files:
        subject_id = file.replace("_synced.csv", "")
        windows = preprocess_subject(f"{input_path}{file}", subject_id)
        all_windows.extend(windows)

    print(f"\nTotal windows: {len(all_windows)}")

    # Save as numpy arrays
    np.save(f"{output_path}windows.npy", all_windows, allow_pickle=True)
    print(f"Saved to {output_path}windows.npy")


if __name__ == "__main__":
    main()
