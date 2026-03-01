"""
feature_engineering.py
-----------------------
Production-ready feature engineering pipeline for the Zomato-like
Smart ETA & Delay Risk Engine dataset.

Input:  data/raw_orders.csv
Output: data/processed_orders.csv
"""

import sys
import pandas as pd


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """Load raw orders CSV and perform basic validation."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        sys.exit(1)

    required_columns = [
        "order_id", "restaurant_id", "rider_id", "city", "order_hour",
        "prep_time_minutes", "distance_km", "traffic_index", "weather_score",
        "rider_experience_years", "predicted_eta_minutes",
        "actual_delivery_time_minutes", "cancelled"
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"[ERROR] Missing expected columns: {missing}")
        sys.exit(1)

    print(f"[INFO] Loaded {len(df):,} rows from '{filepath}'")
    return df


# ---------------------------------------------------------------------------
# Type casting
# ---------------------------------------------------------------------------

def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all numeric columns are properly typed."""
    int_cols = ["order_id", "order_hour", "prep_time_minutes",
                "traffic_index", "cancelled"]
    float_cols = ["distance_km", "weather_score", "rider_experience_years",
                  "predicted_eta_minutes", "actual_delivery_time_minutes"]

    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_delivery_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    delivery_time: alias for actual_delivery_time_minutes.
    Acts as the primary target / reference column for downstream modeling.
    """
    df["delivery_time"] = df["actual_delivery_time_minutes"]
    return df


def add_eta_error(df: pd.DataFrame) -> pd.DataFrame:
    """
    eta_error: difference between actual delivery time and predicted ETA.
    Positive  → delivery was later than predicted (under-estimated).
    Negative  → delivery was faster than predicted (over-estimated).
    """
    df["eta_error"] = (
        df["actual_delivery_time_minutes"] - df["predicted_eta_minutes"]
    )
    return df


def add_is_delayed(df: pd.DataFrame) -> pd.DataFrame:
    """
    is_delayed: binary flag.
    1 → delivery took more than 45 minutes (considered a delay).
    0 → delivered within the acceptable window.
    """
    df["is_delayed"] = (df["delivery_time"] > 45).astype(int)
    return df


def add_is_peak_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    is_peak_hour: binary flag for lunch (12–14) and dinner (19–22) rush hours.
    1 → order placed during peak traffic period.
    0 → off-peak order.
    """
    peak_mask = (
        df["order_hour"].between(12, 14) |
        df["order_hour"].between(19, 22)
    )
    df["is_peak_hour"] = peak_mask.astype(int)
    return df


# ---------------------------------------------------------------------------
# Data quality checks
# ---------------------------------------------------------------------------

def enforce_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove or fix any rows/values that violate data integrity rules:
      - Drop rows with NaN in critical numeric columns.
      - Clip delivery times and ETAs to non-negative values.
      - Ensure eta_error is finite.
    """
    critical_cols = [
        "actual_delivery_time_minutes", "predicted_eta_minutes",
        "order_hour", "cancelled"
    ]

    before = len(df)
    df = df.dropna(subset=critical_cols)
    dropped = before - len(df)
    if dropped:
        print(f"[WARN] Dropped {dropped} rows containing NaN in critical columns.")

    # Clip times to non-negative
    df["delivery_time"] = df["delivery_time"].clip(lower=0)
    df["predicted_eta_minutes"] = df["predicted_eta_minutes"].clip(lower=0)
    df["prep_time_minutes"] = df["prep_time_minutes"].clip(lower=0)
    df["distance_km"] = df["distance_km"].clip(lower=0)
    df["rider_experience_years"] = df["rider_experience_years"].clip(lower=0, upper=5)
    df["weather_score"] = df["weather_score"].clip(lower=0.0, upper=1.0)
    df["traffic_index"] = df["traffic_index"].clip(lower=1, upper=5)

    # Re-derive eta_error after any clipping to stay consistent
    df["eta_error"] = (
        df["actual_delivery_time_minutes"] - df["predicted_eta_minutes"]
    )

    # Fill any remaining NaNs in derived columns with sensible defaults
    df["is_delayed"] = df["is_delayed"].fillna(0).astype(int)
    df["is_peak_hour"] = df["is_peak_hour"].fillna(0).astype(int)
    df["eta_error"] = df["eta_error"].fillna(0.0)

    return df


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Persist the processed DataFrame to CSV."""
    try:
        df.to_csv(filepath, index=False)
        print(f"[INFO] Processed data saved to '{filepath}'")
    except Exception as e:
        print(f"[ERROR] Could not save file: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """Print a concise post-processing summary to stdout."""
    total_rows      = len(df)
    delay_rate      = df["is_delayed"].mean() * 100
    mean_eta_error  = df["eta_error"].mean()
    peak_pct        = df["is_peak_hour"].mean() * 100
    cancel_rate     = df["cancelled"].mean() * 100

    print("\n" + "=" * 48)
    print("  PROCESSING SUMMARY")
    print("=" * 48)
    print(f"  Total rows processed   : {total_rows:,}")
    print(f"  Delay rate (>45 min)   : {delay_rate:.2f}%")
    print(f"  Mean ETA error         : {mean_eta_error:+.2f} minutes")
    print(f"  Peak hour orders       : {peak_pct:.2f}%")
    print(f"  Cancellation rate      : {cancel_rate:.2f}%")
    print("=" * 48 + "\n")


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(input_path: str, output_path: str) -> None:
    """Execute the full feature engineering pipeline end-to-end."""

    # 1. Load raw data
    df = load_data(input_path)

    # 2. Cast all columns to correct numeric types
    df = cast_types(df)

    # 3. Derive new features
    df = add_delivery_time(df)
    df = add_eta_error(df)
    df = add_is_delayed(df)
    df = add_is_peak_hour(df)

    # 4. Enforce data quality constraints
    df = enforce_data_quality(df)

    # 5. Save processed dataset
    save_data(df, output_path)

    # 6. Print summary statistics
    print_summary(df)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    INPUT_PATH  = "data/raw_orders.csv"
    OUTPUT_PATH = "data/raw_orders_v3_realistic.csv"

    run_pipeline(INPUT_PATH, OUTPUT_PATH)