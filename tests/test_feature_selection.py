"""Tests for IV calculation and feature selection."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feature_selection import (
    calculate_iv,
    calculate_vif,
    prune_by_correlation,
    prune_by_vif,
    select_by_iv,
)


def test_iv_perfect_separation():
    """IV on perfectly separating feature returns high IV."""
    # Feature perfectly predicts target
    df = pd.DataFrame({
        "perfect": [0] * 50 + [1] * 50,
        "target": [0] * 50 + [1] * 50
    })
    
    iv_df = calculate_iv(df, "target", ["perfect"])
    assert iv_df.iloc[0]["iv"] > 0.5, f"Perfect separation should give high IV, got {iv_df.iloc[0]['iv']}"


def test_iv_random_noise():
    """IV on random noise feature (binned) returns near-zero IV."""
    np.random.seed(42)
    n = 10000
    # Use binned feature (simulating WOE-transformed feature)
    df = pd.DataFrame({
        "noise_binned": pd.qcut(np.random.randn(n), q=10, duplicates="drop"),
        "target": np.random.binomial(1, 0.5, n)
    })
    
    iv_df = calculate_iv(df, "target", ["noise_binned"])
    assert iv_df.iloc[0]["iv"] < 0.02, f"Random noise should give near-zero IV, got {iv_df.iloc[0]['iv']}"


def test_select_by_iv():
    """IV filtering selects features in range."""
    iv_df = pd.DataFrame({
        "feature": ["a", "b", "c", "d"],
        "iv": [0.01, 0.1, 0.3, 0.6]
    })
    
    selected = select_by_iv(iv_df, iv_min=0.02, iv_max=0.5)
    assert selected == ["b", "c"], f"Expected ['b', 'c'], got {selected}"


def test_prune_by_correlation():
    """Highly correlated features are pruned."""
    # Create correlated features
    np.random.seed(42)
    n = 1000
    base = np.random.randn(n)
    df = pd.DataFrame({
        "f1": base,
        "f2": base + np.random.randn(n) * 0.01,  # ~1.0 correlation
        "f3": np.random.randn(n),  # independent
    })
    
    selected = prune_by_correlation(df, threshold=0.8)
    # Should keep f1 and f3, drop f2 (or vice versa)
    assert len(selected) == 2
    assert "f3" in selected


def test_calculate_vif():
    """VIF calculation works."""
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.randn(n),
    })
    # Make f3 = f1 + f2 (perfect multicollinearity)
    df["f3"] = df["f1"] + df["f2"]
    
    vif_df = calculate_vif(df, ["f1", "f2", "f3"])
    
    # f3 should have very high VIF
    f3_vif = vif_df[vif_df["feature"] == "f3"]["vif"].values[0]
    assert f3_vif > 100, f"Expected high VIF for collinear feature, got {f3_vif}"


def test_prune_by_vif():
    """VIF pruning removes collinear features."""
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.randn(n),
        "f4": np.random.randn(n),
    })
    df["f3"] = df["f1"] + df["f2"]  # collinear
    df["f4"] = df["f1"] * 2  # collinear
    
    selected = prune_by_vif(df, ["f1", "f2", "f3", "f4"], vif_threshold=10)
    
    # Should reduce feature count (at least 2 removed)
    assert len(selected) <= 2
    # All remaining should have VIF < threshold
    vif_df = calculate_vif(df, selected)
    assert (vif_df["vif"] < 10).all()


def test_iv_calculation_handles_constant():
    """IV handles constant feature gracefully."""
    df = pd.DataFrame({
        "constant": [1] * 100,
        "target": [0] * 50 + [1] * 50
    })
    
    iv_df = calculate_iv(df, "target", ["constant"])
    # Constant feature should have IV = 0
    assert iv_df.iloc[0]["iv"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])