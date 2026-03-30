"""
Remove rows with Before_Data_Collection or After_Data_Collection from synced CSVs.
"""

import os
import polars as pl

input_path = "/scratch/bates.car/datasets/paaws_fl_synced/"
output_path = "/scratch/bates.car/datasets/paaws_fl_trimmed/"

os.makedirs(output_path, exist_ok=True)

for file in os.listdir(input_path):
    if not file.endswith(".csv"):
        continue

    print(f"Processing {file}...")

    df = pl.read_csv(f"{input_path}{file}")
    before = len(df)

    df = df.filter(
        ~pl.col("Activity").is_in(["Before_Data_Collection", "After_Data_Collection"])
    ).with_columns(
        pl.when(pl.col("Activity") == "PA_Type_Video_Unavailable/Indecipherable")
        .then(pl.lit("Unknown"))
        .otherwise(pl.col("Activity"))
        .alias("Activity")
    )

    after = len(df)
    print(f"  {before:,} -> {after:,} rows (removed {before - after:,})")

    df.write_csv(f"{output_path}{file}")

print("Done.")
