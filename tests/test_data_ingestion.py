"""Tests for Home Credit data ingestion."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_ingestion import clean_application_train


def test_clean_drops_sk_id_curr():
    """clean_application_train drops SK_ID_CURR column."""
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3],
        "TARGET": [0, 1, 0],
        "DAYS_BIRTH": [-10000, -15000, -20000]
    })
    result = clean_application_train(df)
    assert "SK_ID_CURR" not in result.columns


def test_clean_replaces_days_employed_anomaly():
    """clean_application_train replaces DAYS_EMPLOYED=365243 with NaN."""
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3],
        "TARGET": [0, 1, 0],
        "DAYS_BIRTH": [-10000, -15000, -20000],
        "DAYS_EMPLOYED": [365243, -2000, -5000]  # 365243 is anomaly
    })
    result = clean_application_train(df)
    assert pd.isna(result.loc[0, "DAYS_EMPLOYED"])
    assert result.loc[1, "DAYS_EMPLOYED"] == -2000
    assert result.loc[2, "DAYS_EMPLOYED"] == -5000


def test_clean_drops_xna_gender():
    """clean_application_train drops CODE_GENDER='XNA' rows."""
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3],
        "TARGET": [0, 1, 0],
        "DAYS_BIRTH": [-10000, -15000, -20000],
        "CODE_GENDER": ["M", "XNA", "F"]
    })
    result = clean_application_train(df)
    assert len(result) == 2
    assert "XNA" not in result["CODE_GENDER"].values


def test_clean_handles_other_anomalies():
    """clean_application_train handles DAYS_LAST_PHONE_CHANGE=0 and REGION_RATING=-1."""
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "TARGET": [0, 1],
        "DAYS_BIRTH": [-10000, -15000],
        "DAYS_LAST_PHONE_CHANGE": [0, -1000],
        "REGION_RATING_CLIENT": [-1, 2],
        "REGION_RATING_CLIENT_W_CITY": [-1, 1]
    })
    result = clean_application_train(df)
    assert pd.isna(result.loc[0, "DAYS_LAST_PHONE_CHANGE"])
    assert pd.isna(result.loc[0, "REGION_RATING_CLIENT"])
    assert pd.isna(result.loc[0, "REGION_RATING_CLIENT_W_CITY"])
    assert result.loc[1, "DAYS_LAST_PHONE_CHANGE"] == -1000
    assert result.loc[1, "REGION_RATING_CLIENT"] == 2


def test_clean_target_is_int():
    """clean_application_train ensures TARGET is int."""
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "TARGET": [0.0, 1.0],
        "DAYS_BIRTH": [-10000, -15000]
    })
    result = clean_application_train(df)
    assert result["TARGET"].dtype == int


if __name__ == "__main__":
    pytest.main([__file__, "-v"])