"""
Enhanced feature extraction script for HAR Random Forest pipeline.
Extracts hand-crafted features from 10-second windows.

New features added:
- Signal entropy (spectral and time-domain)
- Zero-crossing rate
- Interquartile range
- Root mean square (RMS)
- Peak-to-peak values
"""

import os
import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert

INPUT_PATH = "PAAWS_FreeLiving_preprocessed/"
OUTPUT_PATH = "PAAWS_FreeLiving_features_v2/"
TARGET_HZ = 30


def time_domain_features(axis: np.ndarray) -> dict:
    """Calculate mean, std, range for a single axis."""
    return {
        "mean": np.mean(axis),
        "std": np.std(axis),
        "range": np.max(axis) - np.min(axis),
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

    yf = fft(signal)
    xf = fftfreq(n, 1 / sampling_rate)

    positive_mask = xf > 0
    xf_pos = xf[positive_mask]
    power = np.abs(yf[positive_mask]) ** 2

    top_indices = np.argsort(power)[-2:][::-1]

    return {
        "dom_freq_1": xf_pos[top_indices[0]] if len(top_indices) > 0 else 0,
        "dom_freq_2": xf_pos[top_indices[1]] if len(top_indices) > 1 else 0
    }


def entropy_features(signal: np.ndarray) -> dict:
    """Calculate time and spectral entropy."""
    # Time-domain entropy (approximate)
    signal_norm = np.abs(signal) / (np.sum(np.abs(signal)) + 1e-10)
    signal_entropy = -np.sum(signal_norm * np.log2(signal_norm + 1e-10))

    # Spectral entropy
    n = len(signal)
    yf = fft(signal)
    power = np.abs(yf[:n//2]) ** 2
    power_norm = power / (np.sum(power) + 1e-10)
    spectral_entropy = -np.sum(power_norm * np.log2(power_norm + 1e-10))

    return {
        "signal_entropy": signal_entropy,
        "spectral_entropy": spectral_entropy
    }


def zero_crossing_rate(signal: np.ndarray) -> dict:
    """Calculate zero-crossing rate."""
    # Count sign changes
    zcr = np.sum(np.abs(np.diff(np.sign(signal)))) / (2 * len(signal))
    return {"zcr": zcr}


def rms_features(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    """Calculate root mean square for each axis and magnitude."""
    mag = np.sqrt(x**2 + y**2 + z**2)
    return {
        "x_rms": np.sqrt(np.mean(x**2)),
        "y_rms": np.sqrt(np.mean(y**2)),
        "z_rms": np.sqrt(np.mean(z**2)),
        "mag_rms": np.sqrt(np.mean(mag**2))
    }


def iqr_features(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    """Calculate interquartile range for each axis and magnitude."""
    mag = np.sqrt(x**2 + y**2 + z**2)
    return {
        "x_iqr": stats.iqr(x),
        "y_iqr": stats.iqr(y),
        "z_iqr": stats.iqr(z),
        "mag_iqr": stats.iqr(mag)
    }


def peak_to_peak_features(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    """Calculate peak-to-peak amplitude for each axis."""
    mag = np.sqrt(x**2 + y**2 + z**2)
    return {
        "x_peak_to_peak": np.max(x) - np.min(x),
        "y_peak_to_peak": np.max(y) - np.min(y),
        "z_peak_to_peak": np.max(z) - np.min(z),
        "mag_peak_to_peak": np.max(mag) - np.min(mag)
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

    # NEW: Entropy features
    for axis_name, axis_data in [("x", x), ("y", y), ("z", z)]:
        ent = entropy_features(axis_data)
        features[f"{axis_name}_entropy"] = ent["signal_entropy"]
        features[f"{axis_name}_spec_entropy"] = ent["spectral_entropy"]

    # NEW: Entropy on magnitude
    mag_ent = entropy_features(magnitude_signal)
    features["mag_entropy"] = mag_ent["signal_entropy"]
    features["mag_spec_entropy"] = mag_ent["spectral_entropy"]

    # NEW: Zero-crossing rate for each axis
    for axis_name, axis_data in [("x", x), ("y", y), ("z", z)]:
        zcr_dict = zero_crossing_rate(axis_data)
        features[f"{axis_name}_zcr"] = zcr_dict["zcr"]

    # NEW: ZCR on magnitude
    mag_zcr = zero_crossing_rate(magnitude_signal)
    features["mag_zcr"] = mag_zcr["zcr"]

    # NEW: RMS features
    rms = rms_features(x, y, z)
    features.update(rms)

    # NEW: IQR features
    iqr = iqr_features(x, y, z)
    features.update(iqr)

    # NEW: Peak-to-peak features
    p2p = peak_to_peak_features(x, y, z)
    features.update(p2p)

    return features


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("Loading windows...")
    windows = np.load(f"{INPUT_PATH}windows.npy", allow_pickle=True)
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
    np.save(f"{OUTPUT_PATH}features.npy", features_list, allow_pickle=True)
    print(f"Saved to {OUTPUT_PATH}features.npy")

    # Also save as CSV for inspection
    import polars as pl
    df = pl.DataFrame(features_list)
    df.write_csv(f"{OUTPUT_PATH}features.csv")
    print(f"Saved to {OUTPUT_PATH}features.csv")

    # Print feature summary
    print(f"\nFeature columns ({len(df.columns)} total): {df.columns}")
    print(f"Label distribution:\n{df.group_by('label').len().sort('len', descending=True)}")


if __name__ == "__main__":
    main()
