import os
import glob
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data(raw_dir):
    """Load and concatenate all raw CSV files."""
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["source_file"] = os.path.basename(f)
        dfs.append(df)
        print(f"  Loaded {os.path.basename(f)}: {len(df)} rows")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows: {len(combined)}")
    print(f"Good posture (1): {(combined['label'] == 1).sum()}")
    print(f"Bad posture  (0): {(combined['label'] == 0).sum()}")

    return combined


def clean_data(df):
    """Clean the raw data."""
    print("\n--- Cleaning ---")
    initial_rows = len(df)

    # 1. Drop rows with any NaN
    df = df.dropna()
    dropped_nan = initial_rows - len(df)
    print(f"Dropped {dropped_nan} rows with NaN values")

    # 2. Drop exact duplicate rows (excluding source_file)
    feature_cols = [c for c in df.columns if c not in ["source_file"]]
    before_dedup = len(df)
    df = df.drop_duplicates(subset=feature_cols)
    dropped_dup = before_dedup - len(df)
    print(f"Dropped {dropped_dup} duplicate rows")

    # 3. Remove extreme outliers using IQR on angle features
    angle_cols = [c for c in df.columns if c not in ["label", "source_file"]]
    before_outlier = len(df)

    outlier_mask = pd.Series([False] * len(df), index=df.index)
    for col in angle_cols:
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        col_outliers = (df[col] < q1) | (df[col] > q99)
        outlier_mask = outlier_mask | col_outliers

    df = df[~outlier_mask]
    dropped_outlier = before_outlier - len(df)
    print(f"Dropped {dropped_outlier} outlier rows (outside 1st-99th percentile)")

    print(f"\nAfter cleaning: {len(df)} rows (removed {initial_rows - len(df)} total)")
    print(f"Good posture (1): {(df['label'] == 1).sum()}")
    print(f"Bad posture  (0): {(df['label'] == 0).sum()}")

    return df


def split_data(df, seed=42, val_size=0.15, test_size=0.15):
    """Stratified train/val/test split."""
    print("\n--- Splitting ---")

    random.seed(seed)
    np.random.seed(seed)

    feature_cols = [c for c in df.columns if c not in ["label", "source_file"]]
    X = df[feature_cols]
    y = df["label"]

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Second split: train vs val
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction, random_state=seed, stratify=y_trainval
    )

    print(f"Train: {len(X_train)} rows ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  Good: {(y_train == 1).sum()}, Bad: {(y_train == 0).sum()}")
    print(f"Val:   {len(X_val)} rows ({len(X_val)/len(df)*100:.1f}%)")
    print(f"  Good: {(y_val == 1).sum()}, Bad: {(y_val == 0).sum()}")
    print(f"Test:  {len(X_test)} rows ({len(X_test)/len(df)*100:.1f}%)")
    print(f"  Good: {(y_test == 1).sum()}, Bad: {(y_test == 0).sum()}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def print_eda(df, feature_cols):
    """Print basic EDA stats."""
    print("\n--- Feature Summary ---")
    print(f"{'Feature':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 60)
    for col in feature_cols:
        print(f"{col:<25} {df[col].mean():>8.3f} {df[col].std():>8.3f} "
              f"{df[col].min():>8.3f} {df[col].max():>8.3f}")

    print("\n--- Feature Means by Class ---")
    print(f"{'Feature':<25} {'Good':>8} {'Bad':>8} {'Diff':>8}")
    print("-" * 52)
    for col in feature_cols:
        good_mean = df[df["label"] == 1][col].mean()
        bad_mean = df[df["label"] == 0][col].mean()
        diff = abs(good_mean - bad_mean)
        print(f"{col:<25} {good_mean:>8.3f} {bad_mean:>8.3f} {diff:>8.3f}")


def run_preprocessing():
    """Full preprocessing pipeline."""

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    output_dir = os.path.join(base_dir, "data", "processed")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("POSTURE DATA PREPROCESSING")
    print("=" * 60)

    # Load
    print("\n--- Loading Raw Data ---")
    df = load_raw_data(raw_dir)

    # Clean
    df = clean_data(df)

    # Get feature columns before dropping source_file
    feature_cols = [c for c in df.columns if c not in ["label", "source_file"]]

    # EDA
    print_eda(df, feature_cols)

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = split_data(df)

    # Save
    print("\n--- Saving ---")

    train_df = X_train.copy()
    train_df["label"] = y_train
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)

    val_df = X_val.copy()
    val_df["label"] = y_val
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)

    test_df = X_test.copy()
    test_df["label"] = y_test
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    # Save feature names
    import json
    with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"Saved train.csv ({len(train_df)} rows)")
    print(f"Saved val.csv ({len(val_df)} rows)")
    print(f"Saved test.csv ({len(test_df)} rows)")
    print(f"Saved feature_names.json ({len(feature_cols)} features)")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_preprocessing()