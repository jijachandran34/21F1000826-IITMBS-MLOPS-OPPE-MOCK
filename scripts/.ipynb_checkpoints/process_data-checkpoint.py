"""
Feature Engineering Script for Iris Dataset
-------------------------------------------
Reads raw CSV data, performs feature engineering, and outputs a processed Parquet file
ready for Feast offline store.

Features created:
- petal_area
- sepal_area
- petal_sepal_ratio
- normalized numeric features
- label-encoded species
- event_timestamp (required by Feast)
- iris_id (entity key for Feast)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features for the Iris dataset."""
    # Compute derived features
    df["petal_area"] = df["petal_length"] * df["petal_width"]
    df["sepal_area"] = df["sepal_length"] * df["sepal_width"]
    df["petal_sepal_ratio"] = np.where(
        df["sepal_area"] != 0,
        df["petal_area"] / df["sepal_area"],
        0,
    )

    # Normalize numeric columns
    num_cols = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "petal_area",
        "sepal_area",
        "petal_sepal_ratio",
    ]
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Encode species (target)
    encoder = LabelEncoder()
    df["species_encoded"] = encoder.fit_transform(df["species"])

    # Add entity column (unique id)
    df["iris_id"] = range(1, len(df) + 1)

    # Add event timestamp (Feast requires this)
    df["event_timestamp"] = datetime.utcnow()

    return df


def main(input_path: str, output_path: str):
    """Read raw CSV, engineer features, and save processed Parquet."""
    print(f"🔹 Reading input data from {input_path}")
    df = pd.read_csv(input_path)

    print("🔹 Performing feature engineering...")
    processed_df = engineer_features(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_parquet(output_path, index=False)
    print(f"✅ Processed data saved to {output_path}")
    print(processed_df.head())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature engineering for Iris dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to save processed Parquet file")
    args = parser.parse_args()

    main(args.input, args.output)
