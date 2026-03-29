"""
Batch processing script for PAAWS FreeLiving dataset.
Uses joblib for parallel processing across subjects.
"""

import os
from joblib import Parallel, delayed
from read_accelerometer_data import data_to_csv

parent_path = "/scratch/bates.car/datasets/paaws_fl/PAAWS_FreeLiving/"
output_path = "/scratch/bates.car/datasets/paaws_fl_synced/"


def process_subject(DS: str) -> str:
    """Process a single subject. Returns status message."""
    accel_path = f"{parent_path}{DS}/accel/{DS}-Free-LeftWrist.csv"
    label_path = f"{parent_path}{DS}/label/{DS}-Free-label.csv"
    out_file = f"{output_path}{DS}_synced.csv"
    
    if not os.path.exists(accel_path):
        return f"[SKIP] {DS}: accel file not found"
    if not os.path.exists(label_path):
        return f"[SKIP] {DS}: label file not found"
    
    try:
        data_to_csv(accel_path, label_path, out_file)
        return f"[OK] {DS}"
    except Exception as e:
        return f"[ERROR] {DS}: {e}"


if __name__ == "__main__":
    os.makedirs(output_path, exist_ok=True)
    
    subjects = os.listdir(parent_path)
    print(f"Found {len(subjects)} subjects")
    
    # Process in parallel (-1 = use all cores)
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_subject)(DS) for DS in subjects
    )
    
    print("\n--- Summary ---")
    for r in results:
        print(r)
