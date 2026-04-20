"""
Experiment 3: Stacking Ensemble
Combines XGBoost, LightGBM, and MLP as base models.
Uses logistic regression as meta-learner.
"""

import os
import numpy as np
import polars as pl
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import warnings
warnings.filterwarnings('ignore')

INPUT_PATH = "PAAWS_FreeLiving_features_v2/"
OUTPUT_PATH = "sams_experiments/results_stacking/"

N_FOLDS = 5
RANDOM_STATE = 42


def load_data():
    print("Loading features...")
    df = pl.read_csv(f"{INPUT_PATH}features.csv")
    meta_cols = ["subject_id", "window_id", "label"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    print(f"Loaded {len(df)} samples, {len(feature_cols)} features")
    return df, feature_cols


def subject_wise_split(subjects: list, fold: int, n_folds: int = N_FOLDS):
    n_subjects = len(subjects)
    fold_size = n_subjects // n_folds
    rotated = subjects[fold * fold_size:] + subjects[:fold * fold_size]
    n_test = max(1, int(n_subjects * 0.2))
    n_val = max(1, int(n_subjects * 0.1))
    test_subjects = rotated[:n_test]
    val_subjects = rotated[n_test:n_test + n_val]
    train_subjects = rotated[n_test + n_val:]
    return train_subjects, val_subjects, test_subjects


def train_and_evaluate(df: pl.DataFrame, feature_cols: list):
    subjects = sorted(df["subject_id"].unique().to_list())
    n_subjects = len(subjects)

    print(f"\n{'='*60}")
    print(f"Starting Stacking CV with {n_subjects} subjects")

    le = LabelEncoder()
    all_labels = df["label"].to_numpy()
    le.fit(all_labels)
    n_classes = len(le.classes_)

    all_preds = []
    all_true = []
    fold_results = []

    for fold in range(N_FOLDS):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

        train_subj, val_subj, test_subj = subject_wise_split(subjects, fold)

        train_df = df.filter(pl.col("subject_id").is_in(train_subj))
        test_df = df.filter(pl.col("subject_id").is_in(test_subj))

        X_train = train_df.select(feature_cols).to_numpy()
        y_train = le.transform(train_df["label"].to_numpy())
        X_test = test_df.select(feature_cols).to_numpy()
        y_test = le.transform(test_df["label"].to_numpy())

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

        raw_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        biking_idx = le.transform(["Biking"])[0]
        sample_weights = np.where(y_train == biking_idx, raw_weights, np.sqrt(raw_weights))
        sample_weights /= sample_weights.mean()

        # Train base models
        print("  Training base models (XGB, LGB, MLP)...")

        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1,
            random_state=RANDOM_STATE, verbosity=0, objective="multi:softmax",
            num_class=n_classes, eval_metric="mlogloss"
        )
        xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

        lgb_model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1,
            random_state=RANDOM_STATE, verbose=-1
        )
        lgb_model.fit(X_train, y_train, sample_weight=sample_weights)

        mlp_model = MLPClassifier(
            hidden_layer_sizes=(128, 64), learning_rate_init=0.001,
            max_iter=500, random_state=RANDOM_STATE
        )
        mlp_model.fit(X_train, y_train, sample_weight=sample_weights)

        # Generate meta-features on test set
        print("  Generating meta-features...")
        xgb_proba = xgb_model.predict_proba(X_test)
        lgb_proba = lgb_model.predict_proba(X_test)
        mlp_proba = mlp_model.predict_proba(X_test)

        X_meta_test = np.hstack([xgb_proba, lgb_proba, mlp_proba])

        # Train meta-learner on training set (using full train set for this demo)
        xgb_proba_train = xgb_model.predict_proba(X_train)
        lgb_proba_train = lgb_model.predict_proba(X_train)
        mlp_proba_train = mlp_model.predict_proba(X_train)
        X_meta_train = np.hstack([xgb_proba_train, lgb_proba_train, mlp_proba_train])

        print("  Training meta-learner...")
        meta_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
        meta_model.fit(X_meta_train, y_train)

        y_pred = meta_model.predict(X_meta_test)

        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"Fold {fold + 1} Macro F1: {f1:.4f}")

        fold_results.append({
            "fold": fold + 1,
            "train_subjects": train_subj,
            "test_subjects": test_subj,
            "f1_macro": f1,
        })

        all_preds.extend(y_pred)
        all_true.extend(y_test)

    print(f"\n{'='*60}")
    print("OVERALL RESULTS - Stacking Ensemble")
    print(f"{'='*60}")

    overall_f1 = f1_score(all_true, all_preds, average='macro')
    print(f"\nMacro F1: {overall_f1:.4f}")

    fold_f1s = [r["f1_macro"] for r in fold_results]
    print(f"Mean Fold F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=le.classes_))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_true, all_preds))

    return fold_results, le


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    df, feature_cols = load_data()
    results, le = train_and_evaluate(df, feature_cols)

    # Train final ensemble
    print("\nTraining final ensemble on all data...")
    X = df.select(feature_cols).to_numpy()
    y = le.transform(df["label"].to_numpy())
    n_classes = len(le.classes_)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    raw_weights = compute_sample_weight(class_weight="balanced", y=y)
    biking_idx = le.transform(["Biking"])[0]
    final_weights = np.where(y == biking_idx, raw_weights, np.sqrt(raw_weights))
    final_weights /= final_weights.mean()

    final_xgb = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1,
        random_state=RANDOM_STATE, verbosity=0, objective="multi:softmax",
        num_class=n_classes, eval_metric="mlogloss"
    )
    final_xgb.fit(X, y, sample_weight=final_weights)

    final_lgb = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1,
        random_state=RANDOM_STATE, verbose=-1
    )
    final_lgb.fit(X, y, sample_weight=final_weights)

    final_mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64), learning_rate_init=0.001,
        max_iter=500, random_state=RANDOM_STATE
    )
    final_mlp.fit(X, y, sample_weight=final_weights)

    X_meta = np.hstack([
        final_xgb.predict_proba(X),
        final_lgb.predict_proba(X),
        final_mlp.predict_proba(X)
    ])

    final_meta = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    final_meta.fit(X_meta, y)

    joblib.dump(final_xgb, f"{OUTPUT_PATH}base_xgb.joblib")
    joblib.dump(final_lgb, f"{OUTPUT_PATH}base_lgb.joblib")
    joblib.dump(final_mlp, f"{OUTPUT_PATH}base_mlp.joblib")
    joblib.dump(final_meta, f"{OUTPUT_PATH}meta_model.joblib")
    joblib.dump(scaler, f"{OUTPUT_PATH}scaler.joblib")
    joblib.dump(le, f"{OUTPUT_PATH}label_encoder.joblib")
    print(f"Ensemble saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
