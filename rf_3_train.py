"""
Random Forest model training for HAR.
5-fold subject-wise cross-validation with 7:1:2 train/val/test split.
"""

import os
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

INPUT_PATH = "/scratch/bates.car/datasets/paaws_fl_features/"
OUTPUT_PATH = "/scratch/bates.car/datasets/paaws_fl_results/"

N_SUBJECTS = 20
N_FOLDS = 5
RANDOM_STATE = 42


def load_data():
    """Load features and prepare for training."""
    print("Loading features...")
    df = pl.read_csv(f"{INPUT_PATH}features.csv")
    
    # Get feature columns (exclude metadata)
    meta_cols = ["subject_id", "window_id", "label"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    print(f"Loaded {len(df)} samples, {len(feature_cols)} features")
    print(f"Subjects: {df['subject_id'].n_unique()}")
    print(f"Labels: {df['label'].unique().to_list()}")
    
    return df, feature_cols


def subject_wise_split(subjects: list, fold: int, n_folds: int = N_FOLDS):
    """
    Split subjects into train/val/test sets for a given fold.
    Ratio: 7:1:2 (train:val:test)
    """
    n_subjects = len(subjects)
    fold_size = n_subjects // n_folds
    
    # Rotate subjects for this fold
    rotated = subjects[fold * fold_size:] + subjects[:fold * fold_size]
    
    # Split: 70% train, 10% val, 20% test
    n_test = max(1, int(n_subjects * 0.2))
    n_val = max(1, int(n_subjects * 0.1))
    n_train = n_subjects - n_test - n_val
    
    test_subjects = rotated[:n_test]
    val_subjects = rotated[n_test:n_test + n_val]
    train_subjects = rotated[n_test + n_val:]
    
    return train_subjects, val_subjects, test_subjects


def held_one_out_split(subjects: list, fold: int):
    """Leave-one-subject-out for small datasets."""
    test_subjects = [subjects[fold]]
    train_subjects = [s for i, s in enumerate(subjects) if i != fold]
    val_subjects = []  # No validation set for LOSO
    return train_subjects, val_subjects, test_subjects


def train_and_evaluate(df: pl.DataFrame, feature_cols: list):
    """Run cross-validation and return results."""
    
    subjects = sorted(df["subject_id"].unique().to_list())
    n_subjects = len(subjects)
    
    print(f"\n{'='*60}")
    print(f"Starting cross-validation with {n_subjects} subjects")
    
    # Choose CV strategy based on dataset size
    if n_subjects >= 10:
        print(f"Using 5-fold subject-wise CV (7:1:2 split)")
        n_folds = N_FOLDS
        split_fn = lambda s, f: subject_wise_split(s, f, N_FOLDS)
    else:
        print(f"Using leave-one-subject-out CV")
        n_folds = n_subjects
        split_fn = held_one_out_split
    
    # Encode labels
    le = LabelEncoder()
    all_labels = df["label"].to_numpy()
    le.fit(all_labels)
    
    all_preds = []
    all_true = []
    all_test_subjects = []
    fold_results = []
    
    for fold in range(n_folds):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        train_subj, val_subj, test_subj = split_fn(subjects, fold)
        print(f"Train: {len(train_subj)} subjects, Val: {len(val_subj)}, Test: {len(test_subj)}")
        
        # Split data
        train_mask = df["subject_id"].is_in(train_subj)
        test_mask = df["subject_id"].is_in(test_subj)
        
        train_df = df.filter(train_mask)
        test_df = df.filter(test_mask)
        
        X_train = train_df.select(feature_cols).to_numpy()
        y_train = le.transform(train_df["label"].to_numpy())
        X_test = test_df.select(feature_cols).to_numpy()
        y_test = le.transform(test_df["label"].to_numpy())
        
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Z-score normalization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Handle NaN/Inf from correlations
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
        
        # Train Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        rf.fit(X_train, y_train)
        
        # Predict
        y_pred = rf.predict(X_test)
        
        # Metrics
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"Fold {fold + 1} Macro F1: {f1:.4f}")
        
        fold_results.append({
            "fold": fold + 1,
            "train_subjects": train_subj,
            "test_subjects": test_subj,
            "f1_macro": f1,
            "n_train": len(X_train),
            "n_test": len(X_test)
        })
        
        all_preds.extend(y_pred)
        all_true.extend(y_test)
        all_test_subjects.extend(test_df["subject_id"].to_list())
    
    # Overall metrics
    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print(f"{'='*60}")
    
    overall_f1 = f1_score(all_true, all_preds, average='macro')
    print(f"\nMacro F1 (all folds): {overall_f1:.4f}")
    
    fold_f1s = [r["f1_macro"] for r in fold_results]
    print(f"Mean Fold F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=le.classes_))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_true, all_preds)
    print(f"Classes: {le.classes_}")
    print(cm)
    
    return fold_results, le.classes_, overall_f1


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    df, feature_cols = load_data()
    
    results, classes, overall_f1 = train_and_evaluate(df, feature_cols)
    
    # Save results (convert subject lists to strings for CSV serialization)
    results_serializable = [
        {**r,
         "train_subjects": ";".join(str(s) for s in r["train_subjects"]),
         "test_subjects": ";".join(str(s) for s in r["test_subjects"])}
        for r in results
    ]
    results_df = pl.DataFrame(results_serializable)
    results_df.write_csv(f"{OUTPUT_PATH}cv_results.csv")
    print(f"\nResults saved to {OUTPUT_PATH}cv_results.csv")
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    le = LabelEncoder()
    X = df.select(feature_cols).to_numpy()
    y = le.fit_transform(df["label"].to_numpy())
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    final_rf = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    final_rf.fit(X, y)
    
    # Save model and scaler
    joblib.dump(final_rf, f"{OUTPUT_PATH}rf_model.joblib")
    joblib.dump(scaler, f"{OUTPUT_PATH}scaler.joblib")
    joblib.dump(le, f"{OUTPUT_PATH}label_encoder.joblib")
    print(f"Model saved to {OUTPUT_PATH}rf_model.joblib")
    
    # Feature importance
    importance = pl.DataFrame({
        "feature": feature_cols,
        "importance": final_rf.feature_importances_
    }).sort("importance", descending=True)
    
    print("\nTop 10 Feature Importances:")
    print(importance.head(10))
    importance.write_csv(f"{OUTPUT_PATH}feature_importance.csv")


if __name__ == "__main__":
    main()