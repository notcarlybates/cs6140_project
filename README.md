# CS6140 Project — HAR Random Forest Pipeline

Human Activity Recognition (HAR) using a Random Forest classifier on PAAWS accelerometer data. The pipeline processes three sensor locations independently: **LeftWrist**, **RightAnkle**, and **RightThigh**.

---

## Data Pipeline

```
paaws_fl_trimmed/<location>/*.csv
        |
        v
[rf_1_preprocess.py]  — resample 80Hz → 30Hz, window into 10s segments
        |
        v
paaws_fl_preprocessed/<location>/windows.npy
        |
        v
[rf_2_features.py]    — extract time-domain, frequency, and magnitude features
        |
        v
paaws_fl_features/<location>/features.{npy,csv}
        |
        v
[rf_3_train.py]       — subject-wise cross-validation, train final RF model
        |
        v
paaws_fl_results/<location>/{cv_results.csv, rf_model.joblib, scaler.joblib,
                              label_encoder.joblib, feature_importance.csv}
```

Each location is processed fully (all 3 steps) before the next location begins.

---

## Running the Pipeline

### Full pipeline via SLURM

```bash
sbatch run_rf_pipeline.sh
```

Loops over all three locations in order: `LeftWrist` → `RightAnkle` → `RightThigh`.

### Run a single step manually

```bash
python rf_1_preprocess.py --location <LOCATION>
python rf_2_features.py   --location <LOCATION>
python rf_3_train.py      --location <LOCATION>
```

---

## Script Arguments

All three scripts share the same required argument:

| Argument | Type | Required | Choices | Description |
|---|---|---|---|---|
| `--location` | `str` | Yes | `LeftWrist`, `RightAnkle`, `RightThigh` | Sensor location to process |

---

## Script Details

### `rf_1_preprocess.py`

Reads raw CSVs from `paaws_fl_trimmed/<location>/`, filters and maps activity labels, resamples from 80Hz to 30Hz, and windows data into 10-second segments.

| Constant | Value | Description |
|---|---|---|
| `TARGET_HZ` | `30` | Target sampling rate after resampling |
| `WINDOW_SEC` | `10` | Window size in seconds |
| `WINDOW_SIZE` | `300` | Samples per window (`TARGET_HZ * WINDOW_SEC`) |
| `ACTIVITY_MAPPING` | `lab_fl_5` | Activity label mapping scheme (see `utils.py`) |

**Output:** `paaws_fl_preprocessed/<location>/windows.npy`

Labels excluded before windowing: `Unknown`, `Before_Data_Collection`, `After_Data_Collection`, and any label not present in the `lab_fl_5` mapping.

---

### `rf_2_features.py`

Loads windowed data and extracts 18 hand-crafted features per window:

- **Time-domain** (per axis X/Y/Z): mean, std, range — 9 features
- **Axis correlations**: corr_xy, corr_xz, corr_yz — 3 features
- **Magnitude**: mean, std, range, MAD, kurtosis, skew — 6 features
- **Frequency** (on magnitude signal): top 2 dominant frequencies — 2 features

| Constant | Value | Description |
|---|---|---|
| `TARGET_HZ` | `30` | Sampling rate used for FFT frequency axis |

**Output:** `paaws_fl_features/<location>/features.npy` and `features.csv`

---

### `rf_3_train.py`

Trains a Random Forest with subject-wise cross-validation.

| Constant | Value | Description |
|---|---|---|
| `N_FOLDS` | `5` | Number of CV folds (used when ≥10 subjects) |
| `RANDOM_STATE` | `42` | Random seed for reproducibility |

**CV strategy:** 5-fold subject-wise with 7:1:2 train/val/test split when ≥10 subjects; leave-one-subject-out otherwise.

**Random Forest hyperparameters** (fixed):

| Parameter | Value |
|---|---|
| `n_estimators` | `100` |
| `max_depth` | `None` (unlimited) |
| `min_samples_split` | `2` |
| `min_samples_leaf` | `1` |
| `n_jobs` | `-1` (all cores) |

**Output:**
- `cv_results.csv` — per-fold F1 scores and subject splits
- `rf_model.joblib` — final model trained on all data
- `scaler.joblib` — fitted StandardScaler
- `label_encoder.joblib` — fitted LabelEncoder
- `feature_importance.csv` — feature importances ranked by mean decrease in impurity

---

## Activity Mapping

Labels are mapped using the `lab_fl_5` scheme from `utils.py`, collapsing raw labels into 5 canonical classes:

| Class | Example raw labels |
|---|---|
| `Sitting` | Sitting_Still, Sit_Typing_Lab, Sit_Writing_Lab, ... |
| `Standing` | Standing_Still, Stand_Conversation_Lab, ... |
| `Lying_Down` | Lying_Still, Lying_On_Back_Lab, ... |
| `Walking` | Walking, Treadmill_3mph_Free_Walk_Lab, ... |
| `Biking` | Stationary_Biking_300_Lab, Cycling_Active_Pedaling_Regular_Bicycle, ... |

---

## Directory Structure

```
cs6140_project/
├── rf_1_preprocess.py
├── rf_2_features.py
├── rf_3_train.py
├── run_rf_pipeline.sh
├── utils.py
└── README.md
```

Scratch data directories (on cluster):

```
/scratch/bates.car/datasets/
├── paaws_fl_trimmed/
│   ├── LeftWrist/
│   ├── RightAnkle/
│   └── RightThigh/
├── paaws_fl_preprocessed/
│   ├── LeftWrist/
│   ├── RightAnkle/
│   └── RightThigh/
├── paaws_fl_features/
│   ├── LeftWrist/
│   ├── RightAnkle/
│   └── RightThigh/
└── paaws_fl_results/
    ├── LeftWrist/
    ├── RightAnkle/
    └── RightThigh/
```
