import polars as pl
df = pl.read_csv("/scratch/bates.car/datasets/paaws_fl_features/features.csv")
print(df.group_by("label").len().sort("len", descending=True))
print(df["label"].unique())
