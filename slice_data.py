"""
Remove rows with Before_Data_Collection or After_Data_Collection from synced CSVs.
"""

import argparse
import os
import polars as pl

BASE_INPUT_PATH = "/scratch/bates.car/datasets/paaws_fl_synced/"
BASE_OUTPUT_PATH = "/scratch/bates.car/datasets/paaws_fl_trimmed/"
LOCATIONS = ["LeftWrist", "RightAnkle", "RightThigh"]

parser = argparse.ArgumentParser(description="Slice synced CSVs for a given sensor location.")
parser.add_argument("--location", required=True, choices=LOCATIONS,
                    help="Sensor location to process (LeftWrist, RightAnkle, RightThigh)")
args = parser.parse_args()
location = args.location

input_path = os.path.join(BASE_INPUT_PATH, location) + "/"
output_path = os.path.join(BASE_OUTPUT_PATH, location) + "/"

print(f"Location:    {location}")
print(f"Input path:  {input_path}")
print(f"Output path: {output_path}")

os.makedirs(input_path, exist_ok=True)
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
