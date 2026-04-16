"""
Experiment 5: Hyperparameter Tuning with Optuna
Systematic Bayesian search over XGBoost hyperparameters.
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

try:
    import optuna
except ImportError:
    print("ERROR: Optuna not installed. Install with: pip install optuna")
    exit(1)


INPUT_PATH = "PAAWS_FreeLiving_features_v2/"
OUTPUT_PATH = "sams_experiments/results_optuna/"

RANDOM_STATE = 42
N_FOLDS = 5
N_TRIALS = 30  # Number of hyperparameter combinations to try


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


def objective(trial, df, feature_cols, le):
    """Objective function for Optuna."""
    # Hyperparameter suggestions
    max_depth = trial.suggest_int("max_depth", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5, log=True)
    n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 5)

    subjects = sorted(df["subject_id"].unique().to_list())
    fold_scores = []

    for fold in range(N_FOLDS):
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

        xgb_clf = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            objective="multi:softmax",
            num_class=len(le.classes_),
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        xgb_clf.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred = xgb_clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        fold_scores.append(f1)

    return np.mean(fold_scores)


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    df, feature_cols = load_data()

    le = LabelEncoder()
    all_labels = df["label"].to_numpy()
    le.fit(all_labels)
    n_classes = len(le.classes_)

    print(f"\n{'='*60}")
    print(f"Starting Optuna hyperparameter search ({N_TRIALS} trials)")
    print(f"{'='*60}\n")

    # Run Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, df, feature_cols, le), n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n{'='*60}")
    print("Best Hyperparameters:")
    print(f"{'='*60}")
    best_params = study.best_params
    pprint_params = {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in best_params.items()}
    for k, v in pprint_params.items():
        print(f"  {k}: {v}")
    print(f"\nBest CV F1 Score: {study.best_value:.4f}")

    # Train final model with best params
    print("\nTraining final model with best hyperparameters...")
    X = df.select(feature_cols).to_numpy()
    y = le.transform(df["label"].to_numpy())

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    raw_weights = compute_sample_weight(class_weight="balanced", y=y)
    biking_idx = le.transform(["Biking"])[0]
    final_weights = np.where(y == biking_idx, raw_weights, np.sqrt(raw_weights))
    final_weights /= final_weights.mean()

    final_xgb = xgb.XGBClassifier(
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        n_estimators=best_params["n_estimators"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        min_child_weight=best_params["min_child_weight"],
        objective="multi:softmax",
        num_class=n_classes,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    final_xgb.fit(X, y, sample_weight=final_weights)

    joblib.dump(final_xgb, f"{OUTPUT_PATH}xgb_tuned_model.joblib")
    joblib.dump(scaler, f"{OUTPUT_PATH}scaler.joblib")
    joblib.dump(le, f"{OUTPUT_PATH}label_encoder.joblib")
    joblib.dump(best_params, f"{OUTPUT_PATH}best_params.joblib")
    print(f"Model saved to {OUTPUT_PATH}")

    # Feature importance
    imp_df = pl.DataFrame({
        "feature": feature_cols,
        "importance": final_xgb.feature_importances_
    }).sort("importance", descending=True)

    print("\nTop 10 Features:")
    print(imp_df.head(10))

    # Save trial history
    trials_df = study.trials_dataframe()
    trials_df.to_csv(f"{OUTPUT_PATH}trial_history.csv", index=False)
    print(f"\nTrial history saved to {OUTPUT_PATH}trial_history.csv")


if __name__ == "__main__":
    main()
