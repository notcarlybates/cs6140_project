"""
Remove rows with Before_Data_Collection or After_Data_Collection from synced CSVs.
"""

import polars as pl
import scratch_io

input_path = "/scratch/bates.car/datasets/paaws_fl_synced/"
output_path = "/scratch/bates.car/datasets/paaws_fl_trimmed/"

scratch_io.makedirs(output_path)

for file in scratch_io.list_dir(input_path):
    if not file.endswith(".csv"):
        continue
    
    print(f"Processing {file}...")
    
    df = scratch_io.read_csv(f"{input_path}{file}")
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
    
    scratch_io.write_csv(df, f"{output_path}{file}")

print("Done.")