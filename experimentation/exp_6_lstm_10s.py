"""
Experiment 6: LSTM Model using 10-second windows
Learns temporal patterns from raw acceleration sequences.
Window size: 10 seconds (300 samples at 30Hz)
"""

import os
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers


    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print('Mixed precision enabled.')
        except Exception as e:
            print('Mixed precision unavailable:', e)
    else:
        print('No GPU detected. Mixed precision disabled.')
        print('For RTX 3070 support on Windows, use WSL2/CUDA or tensorflow-directml.')

except ImportError:
    print("ERROR: TensorFlow not installed. Install with: pip install tensorflow")
    exit(1)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

INPUT_PATH = "PAAWS_FreeLiving_features/"
OUTPUT_PATH = "sams_experiments/results_lstm_10s/"

N_FOLDS = 5
RANDOM_STATE = 42
TARGET_FOLD = 1  

def load_windows():
    """Load raw window sequences for 10-second windows."""
    print("Loading 10-second window sequences...")
    windows = np.load("PAAWS_FreeLiving_preprocessed/windows.npy", allow_pickle=True)
    print(f"Loaded {len(windows)} windows")

    X_list = []
    y_list = []
    subject_ids = []

    for window in windows:
        x = window["X"].astype(np.float32)
        y = window["Y"].astype(np.float32)
        z = window["Z"].astype(np.float32)

        sequence = np.stack([x, y, z], axis=1)
        X_list.append(sequence)
        y_list.append(window["label"])
        subject_ids.append(window["subject_id"])

    X = np.array(X_list)  # (N, 300, 3)
    y = np.array(y_list)

    return X, y, np.array(subject_ids)


def build_lstm_model(input_shape, num_classes):
    """Build LSTM model for sequence classification."""
    model = keras.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),

        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax', dtype='float32')
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
    """Run LSTM cross-validation."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)

    print(f"Classes: {le.classes_}")
    print(f"Input shape: {X.shape}")

    unique_subjects = sorted(set(subject_ids))
    print(f"Total subjects: {len(unique_subjects)}")

    fold_results = []
    all_y_true = []
    all_y_pred = []

    for fold in range(N_FOLDS):
        if fold != TARGET_FOLD:
            continue

        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{N_FOLDS} (TARGET FOLD)")
        print(f"{'='*60}")

        train_subjects, val_subjects, test_subjects = subject_wise_split(unique_subjects, fold)

        train_mask = np.isin(subject_ids, train_subjects)
        val_mask = np.isin(subject_ids, val_subjects)
        test_mask = np.isin(subject_ids, test_subjects)

        X_train, y_train = X[train_mask], y_enc[train_mask]
        X_val, y_val = X[val_mask], y_enc[val_mask]
        X_test, y_test = X[test_mask], y_enc[test_mask]

        print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train.reshape(-1, 3)).reshape(X_train.shape)
        X_val_norm = scaler.transform(X_val.reshape(-1, 3)).reshape(X_val.shape)
        X_test_norm = scaler.transform(X_test.reshape(-1, 3)).reshape(X_test.shape)

        model = build_lstm_model((X_train.shape[1], X_train.shape[2]), num_classes)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("\nTraining...")
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        history = model.fit(
            X_train_norm, y_train,
            validation_data=(X_val_norm, y_val),
            epochs=20, 
            batch_size=128, 
            callbacks=[early_stopping],
            verbose=1
        )

        # Evaluate
        y_pred = model.predict(X_test_norm, verbose=0)
        y_pred_labels = np.argmax(y_pred, axis=1)

        test_acc = accuracy_score(y_test, y_pred_labels)
        test_f1 = f1_score(y_test, y_pred_labels, average='weighted')

        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_labels, target_names=le.classes_))

        fold_results.append({
            'fold': fold,
            'accuracy': test_acc,
            'f1_score': test_f1,
            'model': model,
            'scaler': scaler
        })

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred_labels)

    print(f"\n{'='*60}")
    print(f"RESULTS - Fold {TARGET_FOLD + 1}")
    print(f"{'='*60}")

    if fold_results:
        result = fold_results[0]
        overall_acc = result['accuracy']
        overall_f1 = result['f1_score']

        print(f"Test Accuracy: {overall_acc:.4f}")
        print(f"Test F1-Score: {overall_f1:.4f}")

        os.makedirs(OUTPUT_PATH, exist_ok=True)
        model_path = os.path.join(OUTPUT_PATH, f"lstm_fold{result['fold']}.h5")
        result['model'].save(model_path)
        print(f"\nModel saved to: {model_path}")

        results_text = f"""LSTM Model Results (10-second windows, Fold {TARGET_FOLD + 1})
========================================
Test Accuracy: {overall_acc:.4f}
Test F1-Score: {overall_f1:.4f}
"""
        with open(os.path.join(OUTPUT_PATH, "results_summary.txt"), "w") as f:
            f.write(results_text)

        print(results_text)


if __name__ == "__main__":
    print("Loading data...")
    X, y, subject_ids = load_windows()

    print("\nTraining LSTM model...")
    train_and_evaluate(X, y, subject_ids)
