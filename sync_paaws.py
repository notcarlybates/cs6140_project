"""
Batch processing script for PAAWS FreeLiving dataset.
Uses joblib for parallel processing across subjects.
"""

import argparse
import os
from joblib import Parallel, delayed
from read_accelerometer_data import data_to_csv

PARENT_PATH = "/scratch/bates.car/datasets/paaws_fl/PAAWS_FreeLiving/"
BASE_OUTPUT_PATH = "/scratch/bates.car/datasets/paaws_fl_synced/"
LOCATIONS = ["LeftWrist", "RightAnkle", "RightThigh"]


def process_subject(DS: str, location: str, output_path: str) -> str:
    """Process a single subject. Returns status message."""
    accel_path = f"{PARENT_PATH}{DS}/accel/{DS}-Free-{location}.csv"
    label_path = f"{PARENT_PATH}{DS}/label/{DS}-Free-label.csv"
    out_file = f"{output_path}{DS}_synced.csv"

    if not os.path.exists(accel_path):
        return f"[SKIP] {DS}: accel file not found ({accel_path})"
    if not os.path.exists(label_path):
        return f"[SKIP] {DS}: label file not found"

    try:
        data_to_csv(accel_path, label_path, out_file)
        return f"[OK] {DS}"
    except Exception as e:
        return f"[ERROR] {DS}: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync PAAWS accel data with labels for a given sensor location.")
    parser.add_argument("--location", required=True, choices=LOCATIONS,
                        help="Sensor location to process (LeftWrist, RightAnkle, RightThigh)")
    args = parser.parse_args()
    location = args.location

    output_path = os.path.join(BASE_OUTPUT_PATH, location) + "/"

    print(f"Location:    {location}")
    print(f"Output path: {output_path}")

    os.makedirs(output_path, exist_ok=True)

    subjects = sorted(os.listdir(PARENT_PATH))
    print(f"Found {len(subjects)} subjects")

    # Process in parallel (-1 = use all cores)
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_subject)(DS, location, output_path) for DS in subjects
    )

    print("\n--- Summary ---")
    for r in results:
        print(r)
