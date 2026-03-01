"""
model_training.py
-----------------
Train a RandomForestRegressor to predict food delivery time.

Input:  data/engineered_orders.csv
Output: src/models/saved_model.pkl

Usage:
    python src/models/model_training.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH   = "data/engineered_orders.csv"
MODEL_DIR   = "src/models"
MODEL_PATH  = os.path.join(MODEL_DIR, "saved_model.pkl")

FEATURE_COLS = [
    "prep_time_minutes",
    "distance_km",
    "traffic_index",
    "weather_score",
    "rider_experience_years",
    "is_peak_hour",
]
TARGET_COL = "delivery_time"

RANDOM_STATE = 42
TEST_SIZE    = 0.20


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """Load the engineered orders CSV and validate required columns."""
    if not os.path.exists(filepath):
        print(f"[ERROR] Data file not found: {filepath}")
        sys.exit(1)

    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {len(df):,} rows from '{filepath}'")

    required = FEATURE_COLS + [TARGET_COL]
    missing  = [col for col in required if col not in df.columns]
    if missing:
        print(f"[ERROR] Missing required columns: {missing}")
        sys.exit(1)

    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame):
    """
    Extract feature matrix X and target vector y.
    Drops any rows where features or target contain NaN.
    """
    subset = df[FEATURE_COLS + [TARGET_COL]].copy()

    before = len(subset)
    subset = subset.dropna()
    dropped = before - len(subset)
    if dropped:
        print(f"[WARN] Dropped {dropped} rows with NaN values before training.")

    X = subset[FEATURE_COLS].astype(float)
    y = subset[TARGET_COL].astype(float)

    print(f"[INFO] Feature matrix shape : {X.shape}")
    print(f"[INFO] Target vector shape  : {y.shape}")
    print(f"[INFO] Target mean          : {y.mean():.2f} min")
    print(f"[INFO] Target std           : {y.std():.2f} min")

    return X, y


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_data(X: pd.DataFrame, y: pd.Series):
    """Split into 80/20 train-test sets with a fixed random state."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    print(f"\n[INFO] Train set : {len(X_train):,} rows")
    print(f"[INFO] Test set  : {len(X_test):,} rows")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Fit a RandomForestRegressor on the training data."""
    print("\n[INFO] Training RandomForestRegressor ...")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)
    print("[INFO] Training complete.")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Compute and print regression metrics on the held-out test set:
        - MAE   : Mean Absolute Error
        - RMSE  : Root Mean Squared Error
        - P50   : Median absolute error (50th percentile)
        - P90   : 90th percentile absolute error
    """
    y_pred   = model.predict(X_test)
    abs_err  = np.abs(y_test.values - y_pred)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    p50  = np.percentile(abs_err, 50)
    p90  = np.percentile(abs_err, 90)

    print("\n" + "=" * 50)
    print("  MODEL EVALUATION — TEST SET")
    print("=" * 50)
    print(f"  MAE  (Mean Absolute Error)     : {mae:.3f} min")
    print(f"  RMSE (Root Mean Squared Error) : {rmse:.3f} min")
    print(f"  P50  (Median Absolute Error)   : {p50:.3f} min")
    print(f"  P90  (90th Pct Absolute Error) : {p90:.3f} min")
    print("=" * 50)

    print("\n[INFO] Feature importances:")
    importances = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    for feat, score in importances:
        bar = "█" * int(score * 60)
        print(f"  {feat:<28} {score:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: RandomForestRegressor, path: str) -> None:
    """Serialize the trained model to disk using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"\n[INFO] Model saved to '{path}'  ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_training_pipeline() -> None:
    """Execute the end-to-end model training pipeline."""
    print("=" * 50)
    print("  SMART ETA — MODEL TRAINING PIPELINE")
    print("=" * 50 + "\n")

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Prepare features and target
    X, y = prepare_features(df)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4. Train model
    model = train_model(X_train, y_train)

    # 5. Evaluate on test set
    evaluate_model(model, X_test, y_test)

    # 6. Save model artifact
    save_model(model, MODEL_PATH)

    print("\n[INFO] Pipeline finished successfully.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_training_pipeline()