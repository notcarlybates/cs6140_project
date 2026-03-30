import scratch_io
df = scratch_io.read_csv("/scratch/bates.car/datasets/paaws_fl_features/features.csv")
print(df.group_by("label").len().sort("len", descending=True))
print(df["label"].unique())
