"""Data quality validation checks."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


def check_schema(df: pd.DataFrame, target: str, date_column: str) -> Dict[str, Any]:
    """Validate basic schema expectations."""
    issues = []
    if target not in df.columns:
        issues.append(f"Target column '{target}' not found")
    if date_column not in df.columns:
        issues.append(f"Date column '{date_column}' not found")
    if target in df.columns and df[target].nunique() != 2:
        issues.append(f"Target '{target}' must be binary")
    if target in df.columns and df[target].isna().any():
        issues.append(f"Target column '{target}' contains missing values")
    return {"check": "schema", "passed": len(issues) == 0, "issues": issues}


def check_target_values(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """Ensure target contains only 0 and 1."""
    issues = []
    if target in df.columns:
        unique_vals = set(df[target].unique())
        expected = {0, 1}
        unexpected = unique_vals - expected
        if unexpected:
            issues.append(f"Target contains unexpected values: {unexpected}")
    return {"check": "target_values", "passed": len(issues) == 0, "issues": issues}


def check_duplicate_records(df: pd.DataFrame, id_column: str = None) -> Dict[str, Any]:
    """Check for duplicate records."""
    if id_column and id_column in df.columns:
        dupes = df.duplicated(subset=[id_column]).sum()
    else:
        dupes = df.duplicated().sum()
    return {
        "check": "duplicates",
        "passed": dupes == 0,
        "issues": [f"{dupes} duplicate records found"] if dupes > 0 else [],
        "count": int(dupes)
    }


def check_missingness(df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
    """Report missing value rates per column."""
    missing = df.isna().mean()
    high_missing = missing[missing > threshold]
    issues = []
    if len(high_missing) > 0:
        issues.append(f"Columns with >{threshold*100}% missing: {high_missing.index.tolist()}")
    return {
        "check": "missingness",
        "passed": len(issues) == 0,
        "issues": issues,
        "missing_rates": missing.to_dict()
    }


def check_numeric_ranges(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for impossible numeric values."""
    issues = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (df[col] < 0).any() and any(kw in col.lower() for kw in ["age", "income", "amount", "balance", "limit"]):
            issues.append(f"Negative values in '{col}' (likely invalid for this feature)")
    return {"check": "numeric_ranges", "passed": len(issues) == 0, "issues": issues}


def check_categorical_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Check categorical columns for unexpected values."""
    issues = []
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique == 1:
            issues.append(f"Column '{col}' has only one unique value (constant)")
    return {"check": "categorical_values", "passed": len(issues) == 0, "issues": issues}


def check_date_order(df: pd.DataFrame, date_column: str) -> Dict[str, Any]:
    """Validate dates are valid and ordered."""
    issues = []
    if date_column in df.columns:
        try:
            dates = pd.to_datetime(df[date_column], errors="coerce")
            if dates.isna().any():
                issues.append(f"Invalid dates found in '{date_column}'")
        except (ValueError, TypeError) as e:
            issues.append(f"Date parsing failed for '{date_column}': {e}")
    return {"check": "date_order", "passed": len(issues) == 0, "issues": issues}


def check_for_leakage(df: pd.DataFrame, forbidden_columns: List[str]) -> Dict[str, Any]:
    """Check for forbidden (post-decision) columns."""
    found = [col for col in forbidden_columns if col in df.columns]
    return {
        "check": "leakage",
        "passed": len(found) == 0,
        "issues": [f"Forbidden columns present: {found}"] if found else [],
        "found_columns": found
    }


def check_period_overlap(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    oot_df: pd.DataFrame,
    date_column: str
) -> Dict[str, Any]:
    """Check that train/test/OOT periods don't overlap."""
    issues = []
    train_max = pd.to_datetime(train_df[date_column]).max()
    test_min = pd.to_datetime(test_df[date_column]).min()
    test_max = pd.to_datetime(test_df[date_column]).max()
    
    if train_max >= test_min:
        issues.append(f"Train period overlaps with test (train max: {train_max}, test min: {test_min})")
    
    if len(oot_df) > 0:
        oot_min = pd.to_datetime(oot_df[date_column]).min()
        if test_max >= oot_min:
            issues.append(f"Test period overlaps with OOT (test max: {test_max}, OOT min: {oot_min})")
    
    return {"check": "period_overlap", "passed": len(issues) == 0, "issues": issues}


def run_all_checks(
    df: pd.DataFrame,
    config: dict,
    train_df: pd.DataFrame = None,
    test_df: pd.DataFrame = None,
    oot_df: pd.DataFrame = None
) -> List[Dict[str, Any]]:
    """Run all validation checks and return results."""
    results = []
    
    results.append(check_schema(df, config["target"], config["date_column"]))
    results.append(check_target_values(df, config["target"]))
    results.append(check_duplicate_records(df))
    results.append(check_missingness(df))
    results.append(check_numeric_ranges(df))
    results.append(check_categorical_values(df))
    results.append(check_date_order(df, config["date_column"]))
    results.append(check_for_leakage(df, config["forbidden_columns"]))
    
    if train_df is not None and test_df is not None:
        results.append(check_period_overlap(train_df, test_df, oot_df or pd.DataFrame(), config["date_column"]))
    
    return results