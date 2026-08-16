"""Tests for data preparation: leakage checks, time-based split."""

import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_prep import (
    check_for_leakage, time_based_split, impute_missing,
    check_schema, check_target_values, check_duplicate_records
)


def test_leakage_detection():
    """check_for_leakage raises error when forbidden column present."""
    df = pd.DataFrame({
        "feature": [1, 2, 3],
        "target": [0, 1, 0],
        "recovery_amount": [100, 0, 50]  # forbidden
    })
    
    with pytest.raises(ValueError, match="Leakage detected"):
        check_for_leakage(df, ["recovery_amount"])


def test_leakage_clean():
    """check_for_leakage passes when no forbidden columns."""
    df = pd.DataFrame({
        "feature": [1, 2, 3],
        "target": [0, 1, 0]
    })
    
    # Should not raise
    check_for_leakage(df, ["recovery_amount", "collection_status"])


def test_time_based_split_no_leakage():
    """time_based_split never leaks future dates into train."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "feature": range(100),
        "target": np.random.binomial(1, 0.3, 100)
    })
    
    train, test, oot = time_based_split(df, "date", "2023-04-01", "2023-07-01")
    
    # Train should only have dates before split
    assert (train["date"] < "2023-04-01").all()
    
    # Test should have dates in [split, oot)
    assert (test["date"] >= "2023-04-01").all()
    assert (test["date"] < "2023-07-01").all()
    
    # OOT should have dates >= oot_start
    assert (oot["date"] >= "2023-07-01").all()


def test_time_based_split_sizes():
    """Split produces non-empty sets."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "feature": range(100),
        "target": [0, 1] * 50
    })
    
    train, test, oot = time_based_split(df, "date", "2023-02-15", "2023-03-15")
    
    assert len(train) > 0
    assert len(test) > 0
    assert len(oot) > 0


def test_impute_missing_fit_transform():
    """Imputer fits on train, applies to test."""
    train = pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0],
        "cat": ["A", "B", "A", np.nan],
        "target": [0, 1, 0, 1]
    })
    
    test = pd.DataFrame({
        "num": [np.nan, 3.0],
        "cat": ["C", "A"],
        "target": [0, 1]
    })
    
    train_imp, imputer = impute_missing(train, fit=True)
    test_imp, _ = impute_missing(test, imputer=imputer)
    
    # Train should have no NaN
    assert not train_imp.isna().any().any()
    
    # Test should have no NaN
    assert not test_imp.isna().any().any()
    
    # Imputer should have values for both columns
    assert "num" in imputer
    assert "cat" in imputer


def test_check_schema():
    """Schema validation catches missing target/date."""
    df = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
    
    with pytest.raises(ValueError, match="Date column"):
        check_schema(df, "target", "date")


def test_check_target_values():
    """Target validation catches non-binary values."""
    df = pd.DataFrame({"target": [0, 1, 2]})
    
    with pytest.raises(ValueError, match="unexpected values"):
        check_target_values(df, "target")


def test_check_duplicate_records():
    """Duplicate detection works."""
    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "feature": [10, 20, 20, 30]
    })
    
    result = check_duplicate_records(df, id_column="id")
    assert result["count"] == 1
    assert not result["passed"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])