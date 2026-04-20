"""
Cross-Validation for Final Model with Best Features and Hyperparameters

This script runs proper 5-fold subject-wise CV on:
- Best 15 features (from Optuna feature importance)
- Best hyperparameters (from Optuna trial 27)

Reports honest CV performance (not train-on-test).
"""

import os
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import warnings
warnings.filterwarnings('ignore')

INPUT_PATH = "PAAWS_FreeLiving_features_v2/"
OUTPUT_PATH = "sams_experiments/final_model_cv/"

N_FOLDS = 5
RANDOM_STATE = 42

# Best hyperparameters from Optuna trial 27
BEST_HYPERPARAMS = {
    "max_depth": 7,
    "learning_rate": 0.010078,
    "n_estimators": 500,
    "subsample": 0.624425,
    "colsample_bytree": 0.710855,
    "min_child_weight": 1,
}

# Top 15 features (selected based on importance from Optuna v2)
BEST_FEATURES = [
    "mag_iqr",
    "mag_mad",
    "x_mean",
    "y_zcr",
    "mag_entropy",
    "z_spec_entropy",
    "z_mean",
    "mag_std",
    "z_entropy",
    "mag_spec_entropy",
    "x_std",
    "x_zcr",
    "corr_xy",
    "y_mean",
    "dom_freq_1",
]


def load_data():
    """Load features and prepare for training."""
    print("Loading features...")
    df = pl.read_csv(f"{INPUT_PATH}features.csv")

    meta_cols = ["subject_id", "window_id", "label"]
    all_feature_cols = [c for c in df.columns if c not in meta_cols]

    print(f"Loaded {len(df)} samples")
    print(f"Subjects: {df['subject_id'].n_unique()}")

    return df, all_feature_cols


def subject_wise_split(subjects: list, fold: int, n_folds: int = N_FOLDS):
    """Split subjects into train/val/test sets for a given fold."""
    n_subjects = len(subjects)
    fold_size = n_subjects // n_folds
    rotated = subjects[fold * fold_size:] + subjects[:fold * fold_size]

    n_test = max(1, int(n_subjects * 0.2))
    n_val = max(1, int(n_subjects * 0.1))

    test_subjects = rotated[:n_test]
    val_subjects = rotated[n_test:n_test + n_val]
    train_subjects = rotated[n_test + n_val:]

    return train_subjects, val_subjects, test_subjects


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    df, all_feature_cols = load_data()

    # Filter to only best features
    feature_cols = [f for f in BEST_FEATURES if f in all_feature_cols]
    missing = [f for f in BEST_FEATURES if f not in all_feature_cols]
    if missing:
        print(f"⚠️  Warning: {len(missing)} features not found: {missing}")

    print(f"\nUsing {len(feature_cols)} features")

    subjects = sorted(df["subject_id"].unique().to_list())
    n_subjects = len(subjects)

    le = LabelEncoder()
    all_labels = df["label"].to_numpy()
    le.fit(all_labels)
    n_classes = len(le.classes_)

    print(f"\n{'='*60}")
    print(f"Starting CV with {n_subjects} subjects, {N_FOLDS} folds")
    print(f"Hyperparameters: max_depth={BEST_HYPERPARAMS['max_depth']}, "
          f"learning_rate={BEST_HYPERPARAMS['learning_rate']:.6f}, "
          f"n_estimators={BEST_HYPERPARAMS['n_estimators']}")
    print(f"{'='*60}\n")

    all_preds = []
    all_true = []
    fold_results = []

    for fold in range(N_FOLDS):
        print(f"--- Fold {fold + 1}/{N_FOLDS} ---")

        train_subj, val_subj, test_subj = subject_wise_split(subjects, fold)
        print(f"Train: {len(train_subj)}, Val: {len(val_subj)}, Test: {len(test_subj)}")

        train_df = df.filter(pl.col("subject_id").is_in(train_subj))
        test_df = df.filter(pl.col("subject_id").is_in(test_subj))

        X_train = train_df.select(feature_cols).to_numpy()
        y_train = le.transform(train_df["label"].to_numpy())
        X_test = test_df.select(feature_cols).to_numpy()
        y_test = le.transform(test_df["label"].to_numpy())

        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

        # Class weights
        raw_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        biking_idx = le.transform(["Biking"])[0]
        sample_weights = np.where(y_train == biking_idx, raw_weights, np.sqrt(raw_weights))
        sample_weights /= sample_weights.mean()

        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            max_depth=BEST_HYPERPARAMS["max_depth"],
            learning_rate=BEST_HYPERPARAMS["learning_rate"],
            n_estimators=BEST_HYPERPARAMS["n_estimators"],
            subsample=BEST_HYPERPARAMS["subsample"],
            colsample_bytree=BEST_HYPERPARAMS["colsample_bytree"],
            min_child_weight=BEST_HYPERPARAMS["min_child_weight"],
            objective="multi:softmax",
            num_class=n_classes,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred = xgb_model.predict(X_test)

        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"Fold {fold + 1} Macro F1: {f1:.4f}\n")

        fold_results.append({
            "fold": fold + 1,
            "train_subjects": train_subj,
            "test_subjects": test_subj,
            "f1_macro": f1,
            "n_train": len(X_train),
            "n_test": len(X_test),
        })

        all_preds.extend(y_pred)
        all_true.extend(y_test)

    print(f"{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}\n")

    overall_f1 = f1_score(all_true, all_preds, average='macro')
    print(f"Macro F1 (all folds): {overall_f1:.4f}")

    fold_f1s = [r["f1_macro"] for r in fold_results]
    print(f"Mean Fold F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=le.classes_))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_true, all_preds))

    # Save results
    results_serializable = [
        {**r,
         "train_subjects": ";".join(str(s) for s in r["train_subjects"]),
         "test_subjects": ";".join(str(s) for s in r["test_subjects"])}
        for r in fold_results
    ]
    results_df = pl.DataFrame(results_serializable)
    results_df.write_csv(f"{OUTPUT_PATH}cv_results.csv")
    print(f"\nResults saved to {OUTPUT_PATH}cv_results.csv")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Model: XGBoost with best features + best hyperparameters")
    print(f"Features: {len(feature_cols)}")
    print(f"CV Strategy: 5-fold subject-wise split (7:1:2 train/val/test)")
    print(f"\n✓ Honest CV Macro F1: {overall_f1:.4f}")
    print(f"✓ This is your expected real-world performance")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
