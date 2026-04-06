"""
Feature extraction script for HAR Random Forest pipeline.
Extracts hand-crafted features from 10-second windows.
"""

import argparse
import os
import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq

BASE_INPUT_PATH = "/scratch/bates.car/datasets/paaws_fl_preprocessed/"
BASE_OUTPUT_PATH = "/scratch/bates.car/datasets/paaws_fl_features/"
LOCATIONS = ["LeftWrist", "RightAnkle", "RightThigh"]
TARGET_HZ = 30


def time_domain_features(axis: np.ndarray) -> dict:
    """Calculate mean, std, range for a single axis."""
    return {
        "mean": np.mean(axis),
        "std": np.std(axis),
        "range": np.max(axis) - np.min(axis)
    }


def axis_correlations(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    """Calculate correlations between axis pairs."""
    return {
        "corr_xy": np.corrcoef(x, y)[0, 1],
        "corr_xz": np.corrcoef(x, z)[0, 1],
        "corr_yz": np.corrcoef(y, z)[0, 1]
    }


def magnitude_features(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    """Calculate Euclidean norm (magnitude) features."""
    mag = np.sqrt(x**2 + y**2 + z**2)
    return {
        "mag_mean": np.mean(mag),
        "mag_std": np.std(mag),
        "mag_range": np.max(mag) - np.min(mag),
        "mag_mad": np.median(np.abs(mag - np.median(mag))),
        "mag_kurtosis": stats.kurtosis(mag),
        "mag_skew": stats.skew(mag)
    }


def frequency_features(signal: np.ndarray, sampling_rate: int = TARGET_HZ) -> dict:
    """Extract top 2 dominant frequencies from power spectrum."""
    n = len(signal)

    # Compute FFT
    yf = fft(signal)
    xf = fftfreq(n, 1 / sampling_rate)

    # Get positive frequencies only
    positive_mask = xf > 0
    xf_pos = xf[positive_mask]
    power = np.abs(yf[positive_mask]) ** 2

    # Find top 2 dominant frequencies
    top_indices = np.argsort(power)[-2:][::-1]

    return {
        "dom_freq_1": xf_pos[top_indices[0]] if len(top_indices) > 0 else 0,
        "dom_freq_2": xf_pos[top_indices[1]] if len(top_indices) > 1 else 0
    }


def extract_features(window: dict) -> dict:
    """Extract all features from a single window."""
    x = window["X"]
    y = window["Y"]
    z = window["Z"]

    features = {
        "subject_id": window["subject_id"],
        "window_id": window["window_id"],
        "label": window["label"]
    }

    # Time-domain features per axis
    for axis_name, axis_data in [("x", x), ("y", y), ("z", z)]:
        td = time_domain_features(axis_data)
        features[f"{axis_name}_mean"] = td["mean"]
        features[f"{axis_name}_std"] = td["std"]
        features[f"{axis_name}_range"] = td["range"]

    # Axis correlations
    corr = axis_correlations(x, y, z)
    features.update(corr)

    # Magnitude features
    mag = magnitude_features(x, y, z)
    features.update(mag)

    # Frequency features (on magnitude signal)
    magnitude_signal = np.sqrt(x**2 + y**2 + z**2)
    freq = frequency_features(magnitude_signal)
    features.update(freq)

    return features


def main():
    parser = argparse.ArgumentParser(description="Extract features for a given sensor location.")
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

    print("Loading windows...")
    windows = np.load(f"{input_path}windows.npy", allow_pickle=True)
    print(f"Loaded {len(windows)} windows")

    print("Extracting features...")
    features_list = []
    for i, window in enumerate(windows):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(windows)} windows")

        try:
            features = extract_features(window)
            features_list.append(features)
        except Exception as e:
            print(f"  Error on window {i}: {e}")
            continue

    print(f"Extracted features from {len(features_list)} windows")

    # Convert to numpy structured array and save
    np.save(f"{output_path}features.npy", features_list, allow_pickle=True)
    print(f"Saved to {output_path}features.npy")

    # Also save as CSV for inspection
    import polars as pl
    df = pl.DataFrame(features_list)
    df.write_csv(f"{output_path}features.csv")
    print(f"Saved to {output_path}features.csv")

    # Print feature summary
    print(f"\nFeature columns: {df.columns}")
    print(f"Label distribution:\n{df.group_by('label').len().sort('len', descending=True)}")


if __name__ == "__main__":
    main()
