"""Data loading, cleaning, time-based splitting, and leakage checks."""


import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(path: str) -> pd.DataFrame:
    """Load raw data from CSV."""
    return pd.read_csv(path)


def check_for_leakage(df: pd.DataFrame, forbidden_columns: list[str]) -> None:
    """Fail loudly if any forbidden (post-decision) columns are present."""
    found = [col for col in forbidden_columns if col in df.columns]
    if found:
        raise ValueError(f"Leakage detected: forbidden columns present in data: {found}")


def check_schema(df: pd.DataFrame, target: str, date_column: str) -> None:
    """Validate basic schema expectations."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in data")
    if df[target].nunique() != 2:
        raise ValueError(f"Target '{target}' must be binary, found {df[target].nunique()} unique values")
    if df[target].isna().any():
        raise ValueError(f"Target column '{target}' contains missing values")


def check_target_values(df: pd.DataFrame, target: str) -> None:
    """Ensure target contains only 0 and 1."""
    unique_vals = set(df[target].unique())
    expected = {0, 1}
    if not unique_vals.issubset(expected):
        raise ValueError(f"Target contains unexpected values: {unique_vals - expected}")


def check_duplicate_records(df: pd.DataFrame, id_column: str | None = None) -> dict:
    """Check for duplicate records. Returns dict with count and issues."""
    dupes = df.duplicated(subset=[id_column]).sum() if id_column and id_column in df.columns else df.duplicated().sum()
    return {
        "count": int(dupes),
        "passed": dupes == 0,
        "issues": [f"{dupes} duplicate records found"] if dupes > 0 else []
    }


def check_missingness(df: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """Report missing value rates per column. Warn if above threshold."""
    missing = df.isna().mean()
    high_missing = missing[missing > threshold]
    if len(high_missing) > 0:
        print(f"Warning: Columns with >{threshold*100}% missing: {high_missing.index.tolist()}")
    return missing


def check_numeric_ranges(df: pd.DataFrame) -> None:
    """Check for impossible numeric values (e.g., negative age, income)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (df[col] < 0).any() and "age" in col.lower():
            print(f"Warning: Negative values in '{col}'")
        if (df[col] < 0).any() and "income" in col.lower():
            print(f"Warning: Negative values in '{col}'")


def check_categorical_values(df: pd.DataFrame) -> None:
    """Check categorical columns for unexpected values."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique == 1:
            print(f"Warning: Column '{col}' has only one unique value")


def check_date_order(df: pd.DataFrame, date_column: str) -> None:
    """Validate dates are valid and ordered."""
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    if df[date_column].isna().any():
        raise ValueError(f"Invalid dates found in '{date_column}'")
    if not df[date_column].is_monotonic_increasing:
        print(f"Info: '{date_column}' is not sorted; sorting for time-based split")


def time_based_split(
    df: pd.DataFrame,
    date_column: str,
    split_date: str,
    oot_start_date: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train, test, and OOT sets.
    
    Train: dates < split_date
    Test: split_date <= dates < oot_start_date (or end of data if no OOT)
    OOT: dates >= oot_start_date (if provided)
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    split_dt = pd.to_datetime(split_date)
    
    train = df[df[date_column] < split_dt].copy()
    
    if oot_start_date:
        oot_dt = pd.to_datetime(oot_start_date)
        test = df[(df[date_column] >= split_dt) & (df[date_column] < oot_dt)].copy()
        oot = df[df[date_column] >= oot_dt].copy()
    else:
        test = df[df[date_column] >= split_dt].copy()
        oot = pd.DataFrame(columns=df.columns)
    
    if len(train) == 0:
        raise ValueError("Train set is empty — check split_date")
    if len(test) == 0:
        raise ValueError("Test set is empty — check split_date")
    
    print(f"Split sizes — Train: {len(train)}, Test: {len(test)}, OOT: {len(oot)}")
    return train, test, oot


def impute_missing(
    df: pd.DataFrame,
    imputer: dict | None = None,
    fit: bool = False
) -> tuple[pd.DataFrame, dict]:
    """
    Impute missing values. Fit on train, apply same to test/OOT.
    
    Strategy:
    - Numeric: median
    - Categorical: mode
    """
    df = df.copy()
    if fit:
        imputer = {}
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    imputer[col] = df[col].median()
                else:
                    imputer[col] = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
    else:
        if imputer is None:
            raise ValueError("Imputer must be provided when fit=False")
    
    for col, value in imputer.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
    
    return df, imputer


def prepare_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Full data preparation pipeline."""
    raw_df = load_data(config["raw_path"])
    
    check_schema(raw_df, config["target"], config["date_column"])
    check_target_values(raw_df, config["target"])
    check_for_leakage(raw_df, config["forbidden_columns"])
    check_duplicate_records(raw_df)
    check_missingness(raw_df)
    check_numeric_ranges(raw_df)
    check_categorical_values(raw_df)
    check_date_order(raw_df, config["date_column"])
    
    train_df, test_df, oot_df = time_based_split(
        raw_df,
        config["date_column"],
        config["split_date"],
        config.get("oot_start_date")
    )
    
    train_df, imputer = impute_missing(train_df, fit=True)
    test_df, _ = impute_missing(test_df, imputer=imputer)
    if len(oot_df) > 0:
        oot_df, _ = impute_missing(oot_df, imputer=imputer)
    
    return train_df, test_df, oot_df, imputer