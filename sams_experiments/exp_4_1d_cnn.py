"""
Experiment 4: 1D Convolutional Neural Network
Learns directly from raw acceleration sequences (300 samples per window).
No feature engineering needed.
"""

import os
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print("ERROR: TensorFlow not installed. Install with: pip install tensorflow")
    exit(1)


INPUT_PATH = "PAAWS_FreeLiving_features/"
OUTPUT_PATH = "sams_experiments/results_1d_cnn/"

N_FOLDS = 5
RANDOM_STATE = 42


def load_windows():
    """Load raw window sequences instead of engineered features."""
    print("Loading window sequences...")
    windows = np.load("PAAWS_FreeLiving_preprocessed/windows.npy", allow_pickle=True)
    print(f"Loaded {len(windows)} windows")

    # Extract sequences and labels
    X_list = []
    y_list = []
    subject_ids = []

    for window in windows:
        x = window["X"].astype(np.float32)
        y = window["Y"].astype(np.float32)
        z = window["Z"].astype(np.float32)

        # Stack into (300, 3) array: 300 timesteps, 3 axes
        sequence = np.stack([x, y, z], axis=1)
        X_list.append(sequence)
        y_list.append(window["label"])
        subject_ids.append(window["subject_id"])

    X = np.array(X_list)  # (N, 300, 3)
    y = np.array(y_list)

    return X, y, np.array(subject_ids)


def build_cnn_model(input_shape, num_classes):
    """Build 1D CNN model."""
    model = keras.Sequential([
        layers.Conv1D(32, 5, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        layers.Conv1D(64, 5, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        layers.Conv1D(128, 5, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


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


def train_and_evaluate(X, y, subject_ids):
    """Run 1D CNN cross-validation."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)

    unique_subjects = sorted(np.unique(subject_ids))
    n_subjects = len(unique_subjects)

    print(f"\n{'='*60}")
    print(f"Starting 1D CNN CV with {n_subjects} subjects")

    all_preds = []
    all_true = []
    fold_results = []

    for fold in range(N_FOLDS):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

        train_subj, val_subj, test_subj = subject_wise_split(unique_subjects, fold)

        train_mask = np.isin(subject_ids, train_subj)
        test_mask = np.isin(subject_ids, test_subj)

        X_train = X[train_mask]
        y_train = y_enc[train_mask]
        X_test = X[test_mask]
        y_test = y_enc[test_mask]

        print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

        # Normalize per-axis
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

        print("Building and training 1D CNN...")
        model = build_cnn_model(input_shape=(300, 3), num_classes=num_classes)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Compute class weights for imbalance
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}

        model.fit(
            X_train_scaled, y_train,
            batch_size=32, epochs=20, verbose=0,
            class_weight=class_weight_dict
        )

        y_pred_proba = model.predict(X_test_scaled, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"Fold {fold + 1} Macro F1: {f1:.4f}")

        fold_results.append({
            "fold": fold + 1,
            "f1_macro": f1,
        })

        all_preds.extend(y_pred)
        all_true.extend(y_test)

        del model
        keras.backend.clear_session()

    print(f"\n{'='*60}")
    print("OVERALL RESULTS - 1D CNN")
    print(f"{'='*60}")

    overall_f1 = f1_score(all_true, all_preds, average='macro')
    print(f"\nMacro F1: {overall_f1:.4f}")

    fold_f1s = [r["f1_macro"] for r in fold_results]
    print(f"Mean Fold F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=le.classes_))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_true, all_preds))

    return le


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    X, y, subject_ids = load_windows()
    le = train_and_evaluate(X, y, subject_ids)

    # Train final model
    print("\nTraining final 1D CNN on all data...")

    le_final = LabelEncoder()
    y_enc = le_final.fit_transform(y)
    num_classes = len(le_final.classes_)

    scaler = StandardScaler()
    X_flat = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)

    final_model = build_cnn_model(input_shape=(300, 3), num_classes=num_classes)
    final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_enc), y=y_enc)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    final_model.fit(X_scaled, y_enc, batch_size=32, epochs=30, verbose=1, class_weight=class_weight_dict)

    final_model.save(f"{OUTPUT_PATH}cnn_model.h5")
    joblib.dump(scaler, f"{OUTPUT_PATH}scaler.joblib")
    joblib.dump(le_final, f"{OUTPUT_PATH}label_encoder.joblib")
    print(f"Model saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
