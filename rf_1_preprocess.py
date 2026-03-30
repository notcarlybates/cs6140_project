"""
Preprocessing script for HAR Random Forest pipeline.
1. Resample accelerometer data to 30Hz
2. Window into 10-second segments
"""

import os
import numpy as np
import polars as pl
from scipy import interpolate

INPUT_PATH = "/scratch/bates.car/datasets/paaws_fl_trimmed/"
OUTPUT_PATH = "/scratch/bates.car/datasets/paaws_fl_preprocessed/"
TARGET_HZ = 30
WINDOW_SEC = 10
WINDOW_SIZE = TARGET_HZ * WINDOW_SEC  # 300 samples per window


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

    # Drop rows with invalid/unknown activity labels before resampling and windowing
    if "Activity" in df.columns:
        before = len(df)
        df = df.filter(
            pl.col("Activity").is_not_null() &
            ~pl.col("Activity").is_in(["Unknown", "Before_Data_Collection", "After_Data_Collection"])
        )
        print(f"  Dropped {before - len(df):,} rows with invalid labels ({len(df):,} remaining)")

    print(f"  Resampling from 80Hz to {TARGET_HZ}Hz...")
    df = resample_to_30hz(df, original_hz=80)
    
    print(f"  Creating {WINDOW_SEC}s windows...")
    windows = create_windows(df, subject_id)
    
    print(f"  Created {len(windows)} windows")
    return windows


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    all_windows = []

    files = sorted([f for f in os.listdir(INPUT_PATH) if f.endswith(".csv")])
    print(f"Found {len(files)} subjects")
    
    for file in files:
        subject_id = file.replace("_synced.csv", "")
        windows = preprocess_subject(f"{INPUT_PATH}{file}", subject_id)
        all_windows.extend(windows)
    
    print(f"\nTotal windows: {len(all_windows)}")
    
    # Save as numpy arrays
    np.save(f"{OUTPUT_PATH}windows.npy", all_windows, allow_pickle=True)
    print(f"Saved to {OUTPUT_PATH}windows.npy")


if __name__ == "__main__":
    main()