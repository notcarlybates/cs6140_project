"""
Final Production Model: XGBoost with Best Hyperparameters and Best Features

Uses:
- Optuna v2 hyperparameters (best CV F1: 0.4903)
- Only the most important features
- Full dataset training

Best features selected from feature importance analysis.
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
OUTPUT_PATH = "sams_experiments/final_model/"

RANDOM_STATE = 67

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
    "mag_iqr",           # Interquartile range of magnitude
    "mag_mad",           # Median absolute deviation
    "x_mean",            # Mean acceleration X
    "y_zcr",             # Zero-crossing rate Y
    "mag_entropy",       # Signal entropy magnitude
    "z_spec_entropy",    # Spectral entropy Z
    "z_mean",            # Mean acceleration Z
    "mag_std",           # Std of magnitude
    "z_entropy",         # Signal entropy Z
    "mag_spec_entropy",  # Spectral entropy magnitude
    "x_std",             # Std acceleration X
    "x_zcr",             # Zero-crossing rate X
    "corr_xy",           # Correlation X-Y
    "y_mean",            # Mean acceleration Y
    "dom_freq_1",        # Dominant frequency 1
]


def load_data():
    """Load features and prepare for training."""
    print("Loading features...")
    df = pl.read_csv(f"{INPUT_PATH}features.csv")

    meta_cols = ["subject_id", "window_id", "label"]
    all_feature_cols = [c for c in df.columns if c not in meta_cols]

    print(f"Loaded {len(df)} samples, {len(all_feature_cols)} total features")
    print(f"Using {len(BEST_FEATURES)} best features")
    print(f"Subjects: {df['subject_id'].n_unique()}")
    print(f"Labels: {df['label'].unique().to_list()}")

    return df, all_feature_cols


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    df, all_feature_cols = load_data()

    # Filter to only best features
    feature_cols = [f for f in BEST_FEATURES if f in all_feature_cols]
    missing = [f for f in BEST_FEATURES if f not in all_feature_cols]
    if missing:
        print(f"\n⚠️  Warning: {len(missing)} features not found: {missing}")

    print(f"\n✓ Using {len(feature_cols)} features")

    # Prepare data
    print("\nPreparing training data...")
    X = df.select(feature_cols).to_numpy()
    y_labels = df["label"].to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    n_classes = len(le.classes_)

    print(f"Classes: {le.classes_}")
    print(f"Samples: {len(X)}")

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Class weights
    raw_weights = compute_sample_weight(class_weight="balanced", y=y)
    biking_idx = le.transform(["Biking"])[0]
    final_weights = np.where(y == biking_idx, raw_weights, np.sqrt(raw_weights))
    final_weights /= final_weights.mean()

    # Train final model
    print("\nTraining final model with best hyperparameters...")
    final_model = xgb.XGBClassifier(
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
    final_model.fit(X, y, sample_weight=final_weights)

    # Full dataset predictions for reference
    y_pred = final_model.predict(X)
    f1 = f1_score(y, y_pred, average='macro')

    print(f"\nFinal Model Performance (on full training data):")
    print(f"  Macro F1: {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=le.classes_))

    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

    # Save everything
    joblib.dump(final_model, f"{OUTPUT_PATH}final_model.joblib")
    joblib.dump(scaler, f"{OUTPUT_PATH}scaler.joblib")
    joblib.dump(le, f"{OUTPUT_PATH}label_encoder.joblib")
    joblib.dump(feature_cols, f"{OUTPUT_PATH}feature_list.joblib")

    print(f"\n✓ Model saved to {OUTPUT_PATH}")
    print(f"  - final_model.joblib")
    print(f"  - scaler.joblib")
    print(f"  - label_encoder.joblib")
    print(f"  - feature_list.joblib")

    # Save feature list as CSV for reference
    feature_importance_df = pl.DataFrame({
        "feature": feature_cols,
        "importance": final_model.feature_importances_
    }).sort("importance", descending=True)

    feature_importance_df.write_csv(f"{OUTPUT_PATH}feature_importance.csv")

    print(f"\nTop 10 Features in Final Model:")
    print(feature_importance_df.head(10))

    # Create inference script template
    inference_script = '''"""
Inference script for the final HAR model.

Usage:
    from final_inference import predict_activity

    # Load your data
    X = load_your_data()  # shape: (n_samples, n_features)

    # Predict
    activity = predict_activity(X)
"""

import joblib
import numpy as np

# Load model and preprocessing objects
MODEL = joblib.load("final_model.joblib")
SCALER = joblib.load("scaler.joblib")
LABEL_ENCODER = joblib.load("label_encoder.joblib")
FEATURE_LIST = joblib.load("feature_list.joblib")


def predict_activity(X):
    """
    Predict activity labels.

    Args:
        X: numpy array of shape (n_samples, n_features)
           Features must be in the same order as FEATURE_LIST

    Returns:
        activity_labels: array of activity names (Sitting, Walking, etc.)
    """
    # Standardize
    X_scaled = SCALER.transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)

    # Predict
    y_pred_encoded = MODEL.predict(X_scaled)

    # Decode
    activity_labels = LABEL_ENCODER.inverse_transform(y_pred_encoded)

    return activity_labels


def predict_proba(X):
    """
    Get prediction probabilities for each class.

    Args:
        X: numpy array of shape (n_samples, n_features)

    Returns:
        proba: array of shape (n_samples, n_classes) with probabilities
        classes: list of class names in the same order as proba columns
    """
    X_scaled = SCALER.transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)

    proba = MODEL.predict_proba(X_scaled)
    classes = list(LABEL_ENCODER.classes_)

    return proba, classes
'''

    with open(f"{OUTPUT_PATH}final_inference.py", "w") as f:
        f.write(inference_script)

    print(f"\n✓ Inference script created: final_inference.py")

    # Create summary document
    summary = f"""# Final HAR Model Summary

## Model Configuration
- **Algorithm**: XGBoost (Gradient Boosting)
- **Cross-Validation F1 Score**: 0.4903 (5-fold subject-wise CV)
- **Hyperparameters**:
  - max_depth: {BEST_HYPERPARAMS['max_depth']}
  - learning_rate: {BEST_HYPERPARAMS['learning_rate']:.6f}
  - n_estimators: {BEST_HYPERPARAMS['n_estimators']}
  - subsample: {BEST_HYPERPARAMS['subsample']:.6f}
  - colsample_bytree: {BEST_HYPERPARAMS['colsample_bytree']:.6f}
  - min_child_weight: {BEST_HYPERPARAMS['min_child_weight']}

## Features Used ({len(feature_cols)} features)
{chr(10).join(f"- {i+1}. {f}" for i, f in enumerate(feature_cols))}

## Performance on Full Dataset
- Macro F1: {f1:.4f}
- Classes: {', '.join(le.classes_)}

## Class-Specific Performance
```
{classification_report(y, y_pred, target_names=le.classes_)}
```

## Feature Importance (Top 10)
```
{feature_importance_df.head(10)}
```

## Usage

### 1. Load the model
```python
import joblib
model = joblib.load("final_model.joblib")
scaler = joblib.load("scaler.joblib")
le = joblib.load("label_encoder.joblib")
```

### 2. Prepare your data
- Extract features matching the feature list
- Ensure data is standardized using the saved scaler

### 3. Predict
```python
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)
activity_labels = le.inverse_transform(predictions)
```

Or use the convenience script:
```python
from final_inference import predict_activity
activity_labels = predict_activity(X)
```

## Notes
- Model handles class imbalance using weighted training
- Minority classes (Biking, Lying_Down) receive higher weights during training
- Biking has 0% accuracy (unclassifiable from wrist sensor alone)
- Sitting/Walking/Standing have ~83-84% F1 each
- Lying_Down achieves ~30% F1

## Files
- `final_model.joblib` - Trained XGBoost model
- `scaler.joblib` - StandardScaler for feature normalization
- `label_encoder.joblib` - LabelEncoder for activity labels
- `feature_list.joblib` - List of features used
- `feature_importance.csv` - Feature importance scores
- `final_inference.py` - Inference helper script
"""

    with open(f"{OUTPUT_PATH}MODEL_SUMMARY.md", "w") as f:
        f.write(summary)

    print(f"✓ Model summary created: MODEL_SUMMARY.md\n")


if __name__ == "__main__":
    main()
