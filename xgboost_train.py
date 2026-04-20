"""
Cross-Validation for Final Model with v1 Features and Best Hyperparameters
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

import matplotlib.pyplot as plt
import seaborn as sns

INPUT_PATH = "PAAWS_FreeLiving_features/"
OUTPUT_PATH = "PAAWS_RF/XGBoost_model/"

N_FOLDS = 5
RANDOM_STATE = 67

BEST_HYPERPARAMS = {
    "max_depth": 7,
    "learning_rate": 0.010078,
    "n_estimators": 500,
    "subsample": 0.624425,
    "colsample_bytree": 0.710855,
    "min_child_weight": 1,
}

USE_ALL_V1_FEATURES = True


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
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    df, all_feature_cols = load_data()

    feature_cols = all_feature_cols

    print(f"\nUsing all {len(feature_cols)} v1 features: {feature_cols}")

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
    cm = confusion_matrix(all_true, all_preds)
    print(cm)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_,
                cbar_kws={'label': 'Number of samples'})
    plt.title(f'Confusion Matrix - v1 Features + Best Hyperparameters\nMacro F1: {overall_f1:.4f}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    heatmap_path = f"{OUTPUT_PATH}confusion_matrix_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix heatmap saved to: {heatmap_path}")

    results_serializable = [
        {**r,
         "train_subjects": ";".join(str(s) for s in r["train_subjects"]),
         "test_subjects": ";".join(str(s) for s in r["test_subjects"])}
        for r in fold_results
    ]
    results_df = pl.DataFrame(results_serializable)
    results_df.write_csv(f"{OUTPUT_PATH}cv_results.csv")
    print(f"\nResults saved to {OUTPUT_PATH}cv_results.csv")

    print(f"\n{'='*60}")
    print("SUMMARY - v1 Features + Best Hyperparameters")
    print(f"{'='*60}")
    print(f"Model: XGBoost with all v1 features + best hyperparameters")
    print(f"Features: {len(feature_cols)} (all v1 features)")
    print(f"CV Strategy: 5-fold subject-wise split (7:1:2 train/val/test)")
    print(f"CV Macro F1: {overall_f1:.4f}")
  
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
