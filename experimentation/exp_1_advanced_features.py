"""
Experiment 1: Advanced Feature Engineering
Adds jerk, energy, wavelet, autocorrelation, and angle features.

New features:
- Jerk (derivative of acceleration)
- Signal energy (time and frequency domain)
- Wavelet features (energy in different frequency bands)
- Autocorrelation peaks
- Angle between axes (orientation)
"""

import os
import numpy as np
import pywt
from scipy import stats
from scipy.fft import fft, fftfreq
import polars as pl

INPUT_PATH = "PAAWS_FreeLiving_preprocessed/"
OUTPUT_PATH = "PAAWS_FreeLiving_features_v3/"
TARGET_HZ = 30


def jerk_features(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    """Calculate jerk (3rd derivative = change in acceleration)."""
    # First derivative (velocity-like)
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)

    # Second derivative (velocity change)
    ddx = np.diff(dx)
    ddy = np.diff(dy)
    ddz = np.diff(dz)

    # Jerk magnitude
    jerk_mag = np.sqrt(ddx**2 + ddy**2 + ddz**2)

    return {
        "jerk_mean": np.mean(jerk_mag),
        "jerk_std": np.std(jerk_mag),
        "jerk_max": np.max(jerk_mag),
    }


def energy_features(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    """Calculate signal energy in time and frequency domain."""
    mag = np.sqrt(x**2 + y**2 + z**2)

    # Time-domain energy
    time_energy = np.sum(mag**2)

    # Frequency-domain energy (Parseval)
    n = len(mag)
    yf = fft(mag)
    freq_energy = np.sum(np.abs(yf[:n//2])**2)

    return {
        "time_energy": time_energy,
        "freq_energy": freq_energy,
        "energy_ratio": freq_energy / (time_energy + 1e-10)  # Distribution
    }


def wavelet_features(signal: np.ndarray, wavelet: str = "db4", level: int = 3) -> dict:
    """Extract energy in different wavelet frequency bands."""
    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)

        # Energy in each band
        energies = [np.sum(c**2) for c in coeffs]

        return {
            "wavelet_cA_energy": energies[-1],  # Approximation
            "wavelet_cD1_energy": energies[-2],  # Detail level 1 (highest freq)
            "wavelet_cD2_energy": energies[-3] if len(energies) > 2 else 0,  # Detail level 2
        }
    except Exception:
        return {"wavelet_cA_energy": 0, "wavelet_cD1_energy": 0, "wavelet_cD2_energy": 0}


def autocorrelation_features(signal: np.ndarray, max_lag: int = 30) -> dict:
    """Find peaks in autocorrelation (periodicity)."""
    # Normalize signal
    signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)

    # Autocorrelation
    acf = np.correlate(signal_norm, signal_norm, mode='full')
    acf = acf[len(acf)//2:]
    acf = acf / acf[0]  # Normalize to [0, 1]

    # Find peaks in first max_lag lags
    acf_truncated = acf[1:min(max_lag, len(acf))]

    # Get peak value (periodicity strength)
    peak_idx = np.argmax(acf_truncated) + 1
    peak_value = acf_truncated[peak_idx - 1] if len(acf_truncated) > 0 else 0

    return {
        "autocorr_peak_lag": peak_idx,
        "autocorr_peak_value": peak_value,
    }


def angle_features(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    """Calculate angles between axes (orientation features)."""
    # Mean direction vectors
    x_mean, y_mean, z_mean = np.mean(x), np.mean(y), np.mean(z)

    # Normalize
    mag_mean = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2) + 1e-10

    # Angles from each axis (in radians)
    angle_x = np.arctan2(np.sqrt(y_mean**2 + z_mean**2), x_mean)
    angle_y = np.arctan2(np.sqrt(x_mean**2 + z_mean**2), y_mean)
    angle_z = np.arctan2(np.sqrt(x_mean**2 + y_mean**2), z_mean)

    return {
        "angle_x": angle_x,
        "angle_y": angle_y,
        "angle_z": angle_z,
    }


def extract_features(window: dict) -> dict:
    """Extract all features including advanced ones."""
    x = window["X"]
    y = window["Y"]
    z = window["Z"]
    mag = np.sqrt(x**2 + y**2 + z**2)

    features = {
        "subject_id": window["subject_id"],
        "window_id": window["window_id"],
        "label": window["label"]
    }

    # Original features (time-domain per axis)
    for axis_name, axis_data in [("x", x), ("y", y), ("z", z)]:
        features[f"{axis_name}_mean"] = np.mean(axis_data)
        features[f"{axis_name}_std"] = np.std(axis_data)
        features[f"{axis_name}_range"] = np.max(axis_data) - np.min(axis_data)

    # Correlations
    features["corr_xy"] = np.corrcoef(x, y)[0, 1]
    features["corr_xz"] = np.corrcoef(x, z)[0, 1]
    features["corr_yz"] = np.corrcoef(y, z)[0, 1]

    # Magnitude features
    features["mag_mean"] = np.mean(mag)
    features["mag_std"] = np.std(mag)
    features["mag_range"] = np.max(mag) - np.min(mag)
    features["mag_mad"] = np.median(np.abs(mag - np.median(mag)))
    features["mag_kurtosis"] = stats.kurtosis(mag)
    features["mag_skew"] = stats.skew(mag)

    # Frequency features
    n = len(mag)
    yf = fft(mag)
    xf = fftfreq(n, 1 / TARGET_HZ)
    positive_mask = xf > 0
    power = np.abs(yf[positive_mask]) ** 2
    top_indices = np.argsort(power)[-2:][::-1]
    features["dom_freq_1"] = xf[np.where(positive_mask)[0][top_indices[0]]] if len(top_indices) > 0 else 0
    features["dom_freq_2"] = xf[np.where(positive_mask)[0][top_indices[1]]] if len(top_indices) > 1 else 0

    # NEW: Jerk features
    jerk = jerk_features(x, y, z)
    features.update(jerk)

    # NEW: Energy features
    energy = energy_features(x, y, z)
    features.update(energy)

    # NEW: Wavelet features (on magnitude signal)
    wavelet = wavelet_features(mag)
    features.update(wavelet)

    # NEW: Autocorrelation features (on magnitude signal)
    autocorr = autocorrelation_features(mag)
    features.update(autocorr)

    # NEW: Angle features
    angle = angle_features(x, y, z)
    features.update(angle)

    return features


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("Loading windows...")
    windows = np.load(f"{INPUT_PATH}windows.npy", allow_pickle=True)
    print(f"Loaded {len(windows)} windows")

    print("Extracting advanced features...")
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

    df = pl.DataFrame(features_list)
    df.write_csv(f"{OUTPUT_PATH}features.csv")
    print(f"\nSaved to {OUTPUT_PATH}features.csv")
    print(f"Total features: {len(df.columns) - 3}")  # -3 for metadata cols
    print(f"Label distribution:\n{df.group_by('label').len().sort('len', descending=True)}")


if __name__ == "__main__":
    main()
