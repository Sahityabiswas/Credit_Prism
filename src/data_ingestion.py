"""Data ingestion for Home Credit Default Risk dataset."""

import os
import pandas as pd
from typing import Optional


def load_application_train() -> pd.DataFrame:
    """Load application_train_dated from Hugging Face datasets."""
    from datasets import load_dataset
    
    ds = load_dataset("mohameddhameem/home-credit-default-risk", "application_train_dated")
    return ds["train"].to_pandas()


def clean_application_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean application_train_dated for modeling.
    
    Cleaning steps:
    - Drop SK_ID_CURR (identifier, not predictive)
    - Replace DAYS_EMPLOYED=365243 with NaN (anomalous code for unemployed)
    - Drop CODE_GENDER="XNA" rows (2 rows, invalid category)
    - Handle other known anomalous values
    """
    df = df.copy()
    
    # Drop identifier
    if "SK_ID_CURR" in df.columns:
        df = df.drop(columns=["SK_ID_CURR"])
    
    # Replace DAYS_EMPLOYED anomaly (365243 = not employed)
    if "DAYS_EMPLOYED" in df.columns:
        df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = pd.NA
    
    # Drop invalid gender rows
    if "CODE_GENDER" in df.columns:
        df = df[df["CODE_GENDER"] != "XNA"].copy()
    
    # Handle other known anomalies
    # DAYS_LAST_PHONE_CHANGE: 0 means unknown
    if "DAYS_LAST_PHONE_CHANGE" in df.columns:
        df.loc[df["DAYS_LAST_PHONE_CHANGE"] == 0, "DAYS_LAST_PHONE_CHANGE"] = pd.NA
    
    # REGION_RATING_CLIENT_W_CITY: -1 means unknown
    for col in ["REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY"]:
        if col in df.columns:
            df.loc[df[col] == -1, col] = pd.NA
    
    # Convert target to int if needed
    if "TARGET" in df.columns:
        df["TARGET"] = df["TARGET"].astype(int)
    
    return df


def save_processed(df: pd.DataFrame, path: str) -> None:
    """Save processed DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def load_or_create_clean_data(
    raw_path: str = "data/processed/application_train_clean.csv",
    force_reload: bool = False
) -> pd.DataFrame:
    """
    Load clean data from cache, or download and clean if not exists.
    
    Args:
        raw_path: Path to cached clean CSV
        force_reload: If True, re-download and re-clean
    
    Returns:
        Clean DataFrame ready for pipeline
    """
    if os.path.exists(raw_path) and not force_reload:
        print(f"Loading cached clean data from {raw_path}")
        return pd.read_csv(raw_path)
    
    print("Downloading Home Credit application_train_dated from Hugging Face...")
    df = load_application_train()
    print(f"  Raw shape: {df.shape}")
    
    print("Cleaning data...")
    df = clean_application_train(df)
    print(f"  Clean shape: {df.shape}")
    
    print(f"Saving to {raw_path}")
    save_processed(df, raw_path)
    
    return df


if __name__ == "__main__":
    # Quick test
    df = load_or_create_clean_data()
    print(f"\nColumns ({len(df.columns)}): {list(df.columns)}")
    print(f"\nTarget distribution:\n{df['TARGET'].value_counts()}")
    print(f"\nMissing values:\n{df.isna().sum().sort_values(ascending=False).head(20)}")