
import polars as pl, os
path = '/scratch/bates.car/datasets/paaws_fl_trimmed/ankle/'
all_labels = set()
for f in os.listdir(path):
    if f.endswith('.csv'):
        df = pl.read_csv(path + f)
        if 'Activity' in df.columns:
            all_labels.update(df['Activity'].drop_nulls().unique().to_list())
print(sorted(all_labels))