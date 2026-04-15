"""
Compare SSL fine-tuning vs Random Forest baseline across locations.

Loads cv_results.csv from both pipelines and prints a side-by-side
summary table (mean ± std for macro F1, subject-wise F1, kappa).

SSL results:  /scratch/bates.car/datasets/paaws_ssl_results/<location>/cv_results.csv
RF results:   /scratch/bates.car/datasets/paaws_fl_results/<location>/cv_results.csv
"""

import os

import numpy as np
import polars as pl

SSL_RESULTS_PATH = "/scratch/bates.car/datasets/paaws_ssl_results"
RF_RESULTS_PATH  = "/scratch/bates.car/datasets/paaws_fl_results"
LOCATIONS        = ["LeftWrist", "RightAnkle", "RightThigh"]


def _mean_std(values: list) -> str:
    a = np.array(values, dtype=float)
    return f"{a.mean():.4f} ± {a.std():.4f}"


def load_ssl(location: str) -> pl.DataFrame | None:
    path = os.path.join(SSL_RESULTS_PATH, location, "cv_results.csv")
    if not os.path.exists(path):
        return None
    return pl.read_csv(path)


def load_rf(location: str) -> pl.DataFrame | None:
    path = os.path.join(RF_RESULTS_PATH, location, "cv_results.csv")
    if not os.path.exists(path):
        return None
    return pl.read_csv(path)


def summarise(location: str) -> dict:
    ssl_df = load_ssl(location)
    rf_df  = load_rf(location)

    row = {"location": location}

    if ssl_df is not None:
        row["ssl_macro_f1"]   = _mean_std(ssl_df["macro_f1"].to_list())
        row["ssl_subject_f1"] = _mean_std(ssl_df["subject_f1"].to_list())
        row["ssl_kappa"]      = _mean_std(ssl_df["kappa"].to_list())
    else:
        row["ssl_macro_f1"] = row["ssl_subject_f1"] = row["ssl_kappa"] = "—"

    if rf_df is not None:
        row["rf_macro_f1"] = _mean_std(rf_df["f1_macro"].to_list())
    else:
        row["rf_macro_f1"] = "—"

    return row


def main():
    rows = [summarise(loc) for loc in LOCATIONS]

    col_w = 22
    loc_w = 12

    header = (
        f"{'Location':<{loc_w}}"
        f"{'SSL macro-F1':^{col_w}}"
        f"{'SSL subject-F1':^{col_w}}"
        f"{'SSL kappa':^{col_w}}"
        f"{'RF macro-F1':^{col_w}}"
    )
    sep = "-" * len(header)

    print()
    print("SSL fine-tuning vs Random Forest — cross-validation results")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['location']:<{loc_w}}"
            f"{r['ssl_macro_f1']:^{col_w}}"
            f"{r['ssl_subject_f1']:^{col_w}}"
            f"{r['ssl_kappa']:^{col_w}}"
            f"{r['rf_macro_f1']:^{col_w}}"
        )
    print(sep)
    print()

    # Check if any location has results for both pipelines
    both = [r for r in rows
            if r["ssl_macro_f1"] != "—" and r["rf_macro_f1"] != "—"]
    if not both:
        print("No locations have results for both pipelines yet.")
        return

    print("Delta (SSL − RF) macro-F1 where both are available:")
    for r in both:
        ssl_val = float(r["ssl_macro_f1"].split("±")[0])
        rf_val  = float(r["rf_macro_f1"].split("±")[0])
        delta   = ssl_val - rf_val
        sign    = "+" if delta >= 0 else ""
        print(f"  {r['location']:<{loc_w}} {sign}{delta:.4f}")
    print()


if __name__ == "__main__":
    main()
