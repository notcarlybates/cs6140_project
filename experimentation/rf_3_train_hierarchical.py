"""
Two-stage hierarchical classifier for HAR.

Stage 1: Binary XGBoost — Lying_Down vs. everything else
Stage 2: Multi-class XGBoost — Sitting / Standing / Walking / Biking
         (only runs when Stage 1 says "not Lying_Down")

This isolates Lying_Down from the main classifier, reducing the large
Lying_Down <-> Sitting confusion seen in the flat model.
"""

import os
import numpy as np
import polars as pl
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import warnings
warnings.filterwarnings('ignore')

INPUT_PATH = "PAAWS_FreeLiving_features/"
OUTPUT_PATH = "PAAWS_FreeLiving_results_hierarchical/"

N_FOLDS = 5
RANDOM_STATE = 42
LYING_DOWN = "Lying_Down"
STAGE1_THRESHOLD = 0.65   # raise above 0.5 to reduce Sitting → Lying_Down false positives


def load_data():
    print("Loading features...")
    df = pl.read_csv(f"{INPUT_PATH}features.csv")
    meta_cols = ["subject_id", "window_id", "label"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    print(f"Loaded {len(df)} samples, {len(feature_cols)} features")
    print(f"Subjects: {df['subject_id'].n_unique()}")
    print(f"Labels: {df['label'].unique().to_list()}")
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


def held_one_out_split(subjects: list, fold: int):
    test_subjects = [subjects[fold]]
    train_subjects = [s for i, s in enumerate(subjects) if i != fold]
    return train_subjects, [], test_subjects


def build_stage1_model():
    """Binary: Lying_Down (1) vs. everything else (0)."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )


def build_stage2_model(n_classes: int):
    """Multi-class: Sitting / Standing / Walking / Biking."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        num_class=n_classes,
        objective="multi:softmax",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )


def train_and_evaluate(df: pl.DataFrame, feature_cols: list):
    subjects = sorted(df["subject_id"].unique().to_list())
    n_subjects = len(subjects)

    print(f"\n{'='*60}")
    print(f"Starting cross-validation with {n_subjects} subjects")

    if n_subjects >= 10:
        print("Using 5-fold subject-wise CV (7:1:2 split)")
        n_folds = N_FOLDS
        split_fn = lambda s, f: subject_wise_split(s, f, N_FOLDS)
    else:
        print("Using leave-one-subject-out CV")
        n_folds = n_subjects
        split_fn = held_one_out_split

    # Stage 2 label encoder (excludes Lying_Down)
    stage2_classes = sorted([c for c in df["label"].unique().to_list() if c != LYING_DOWN])
    le2 = LabelEncoder()
    le2.fit(stage2_classes)

    # Final label encoder covering all classes (for unified reporting)
    le_all = LabelEncoder()
    le_all.fit(sorted(df["label"].unique().to_list()))

    all_preds = []
    all_true = []
    fold_results = []

    for fold in range(n_folds):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        train_subj, val_subj, test_subj = split_fn(subjects, fold)
        print(f"Train: {len(train_subj)} subjects, Val: {len(val_subj)}, Test: {len(test_subj)}")

        train_df = df.filter(pl.col("subject_id").is_in(train_subj))
        test_df  = df.filter(pl.col("subject_id").is_in(test_subj))

        X_train_all = train_df.select(feature_cols).to_numpy()
        X_test_all  = test_df.select(feature_cols).to_numpy()

        # --- Stage 1 ---
        y1_train = (train_df["label"] == LYING_DOWN).cast(pl.Int8).to_numpy()
        y1_test  = (test_df["label"]  == LYING_DOWN).cast(pl.Int8).to_numpy()

        scaler1 = StandardScaler()
        X1_train = scaler1.fit_transform(X_train_all)
        X1_test  = scaler1.transform(X_test_all)
        X1_train = np.nan_to_num(X1_train, nan=0, posinf=0, neginf=0)
        X1_test  = np.nan_to_num(X1_test,  nan=0, posinf=0, neginf=0)

        sw1 = compute_sample_weight(class_weight="balanced", y=y1_train)

        print("  Training Stage 1 (Lying_Down detector)...")
        m1 = build_stage1_model()
        m1.fit(X1_train, y1_train, sample_weight=sw1)

        stage1_proba = m1.predict_proba(X1_test)[:, 1]
        stage1_pred = (stage1_proba >= STAGE1_THRESHOLD).astype(int)

        # --- Stage 2 (train on non-Lying_Down samples only) ---
        mask_train_s2 = train_df["label"] != LYING_DOWN
        train_df_s2 = train_df.filter(mask_train_s2)

        X2_train = train_df_s2.select(feature_cols).to_numpy()
        y2_train = le2.transform(train_df_s2["label"].to_numpy())

        scaler2 = StandardScaler()
        X2_train = scaler2.fit_transform(X2_train)
        X2_train = np.nan_to_num(X2_train, nan=0, posinf=0, neginf=0)

        # Softer weights for Stage 2 minority classes
        raw_w2 = compute_sample_weight(class_weight="balanced", y=y2_train)
        biking_idx = le2.transform(["Biking"])[0]
        sw2 = np.where(y2_train == biking_idx, raw_w2, np.sqrt(raw_w2))
        sw2 /= sw2.mean()

        print("  Training Stage 2 (Sitting / Standing / Walking / Biking)...")
        m2 = build_stage2_model(n_classes=len(stage2_classes))
        m2.fit(X2_train, y2_train, sample_weight=sw2)

        # --- Inference: route through stages ---
        # For Stage-2, run ALL test samples through it then override with Stage-1
        X2_test = scaler2.transform(X_test_all)
        X2_test = np.nan_to_num(X2_test, nan=0, posinf=0, neginf=0)
        stage2_pred_labels = le2.inverse_transform(m2.predict(X2_test))

        # Where Stage 1 says Lying_Down, override Stage 2 output
        final_labels = np.where(stage1_pred == 1, LYING_DOWN, stage2_pred_labels)

        # Encode to unified label space for metrics
        y_true_enc = le_all.transform(test_df["label"].to_numpy())
        y_pred_enc = le_all.transform(final_labels)

        f1 = f1_score(y_true_enc, y_pred_enc, average='macro')
        print(f"  Fold {fold + 1} Macro F1: {f1:.4f}")

        fold_results.append({
            "fold": fold + 1,
            "train_subjects": train_subj,
            "test_subjects": test_subj,
            "f1_macro": f1,
            "n_train": len(X_train_all),
            "n_test": len(X_test_all),
        })

        all_preds.extend(y_pred_enc)
        all_true.extend(y_true_enc)

    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print(f"{'='*60}")

    overall_f1 = f1_score(all_true, all_preds, average='macro')
    print(f"\nMacro F1 (all folds): {overall_f1:.4f}")

    fold_f1s = [r["f1_macro"] for r in fold_results]
    print(f"Mean Fold F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=le_all.classes_))

    print("\nConfusion Matrix:")
    print(f"Classes: {le_all.classes_}")
    print(confusion_matrix(all_true, all_preds))

    return fold_results, le_all.classes_, overall_f1


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    df, feature_cols = load_data()
    results, _, _ = train_and_evaluate(df, feature_cols)

    results_serializable = [
        {**r,
         "train_subjects": ";".join(str(s) for s in r["train_subjects"]),
         "test_subjects": ";".join(str(s) for s in r["test_subjects"])}
        for r in results
    ]
    pl.DataFrame(results_serializable).write_csv(f"{OUTPUT_PATH}cv_results.csv")
    print(f"\nResults saved to {OUTPUT_PATH}cv_results.csv")

    # Train final models on all data
    print("\nTraining final models on all data...")
    le2 = LabelEncoder()
    stage2_classes = sorted([c for c in df["label"].unique().to_list() if c != LYING_DOWN])
    le2.fit(stage2_classes)

    le_all = LabelEncoder()
    le_all.fit(sorted(df["label"].unique().to_list()))

    X_all = df.select(feature_cols).to_numpy()
    y_all_labels = df["label"].to_numpy()

    # Stage 1 final
    y1 = (df["label"] == LYING_DOWN).cast(pl.Int8).to_numpy()
    scaler1 = StandardScaler()
    X1 = scaler1.fit_transform(X_all)
    X1 = np.nan_to_num(X1, nan=0, posinf=0, neginf=0)
    sw1 = compute_sample_weight(class_weight="balanced", y=y1)
    final_m1 = build_stage1_model()
    final_m1.fit(X1, y1, sample_weight=sw1)

    # Stage 2 final
    mask_s2 = df["label"] != LYING_DOWN
    df_s2 = df.filter(mask_s2)
    X2 = df_s2.select(feature_cols).to_numpy()
    y2 = le2.transform(df_s2["label"].to_numpy())
    scaler2 = StandardScaler()
    X2 = scaler2.fit_transform(X2)
    X2 = np.nan_to_num(X2, nan=0, posinf=0, neginf=0)
    raw_w2 = compute_sample_weight(class_weight="balanced", y=y2)
    biking_idx = le2.transform(["Biking"])[0]
    sw2 = np.where(y2 == biking_idx, raw_w2, np.sqrt(raw_w2))
    sw2 /= sw2.mean()
    final_m2 = build_stage2_model(n_classes=len(stage2_classes))
    final_m2.fit(X2, y2, sample_weight=sw2)

    joblib.dump(final_m1,  f"{OUTPUT_PATH}stage1_lying_down.joblib")
    joblib.dump(scaler1,   f"{OUTPUT_PATH}stage1_scaler.joblib")
    joblib.dump(final_m2,  f"{OUTPUT_PATH}stage2_main.joblib")
    joblib.dump(scaler2,   f"{OUTPUT_PATH}stage2_scaler.joblib")
    joblib.dump(le2,       f"{OUTPUT_PATH}stage2_label_encoder.joblib")
    joblib.dump(le_all,    f"{OUTPUT_PATH}label_encoder_all.joblib")
    print(f"Models saved to {OUTPUT_PATH}")

    # Feature importance for both stages
    for name, model, cols in [
        ("stage1", final_m1, feature_cols),
        ("stage2", final_m2, feature_cols),
    ]:
        imp_df = pl.DataFrame({
            "feature": cols,
            "importance": model.feature_importances_
        }).sort("importance", descending=True)
        imp_df.write_csv(f"{OUTPUT_PATH}{name}_feature_importance.csv")
        print(f"\nTop 10 {name} features:")
        print(imp_df.head(10))


if __name__ == "__main__":
    main()
