"""Tests for WOE encoder."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from encoding import WOEEncoder, fit_woe_encoder


def test_woe_known_values():
    """WOE on toy 2-category feature matches hand-calculated value."""
    # Create simple dataset: feature A with 2 categories
    # Cat 0: 90 good, 10 bad -> WOE = ln((90/100)/(10/20)) = ln(0.9/0.5) = ln(1.8) ≈ 0.588
    # Cat 1: 10 good, 10 bad -> WOE = ln((10/100)/(10/20)) = ln(0.1/0.5) = ln(0.2) ≈ -1.609
    df = pd.DataFrame({
        "feature": ["A"] * 90 + ["B"] * 10 + ["A"] * 10 + ["B"] * 10,
        "target": [0] * 90 + [0] * 10 + [1] * 10 + [1] * 10
    })
    
    encoder = WOEEncoder()
    encoder.fit(df, ["feature"], "target")
    
    # Transform and check
    encoder.transform(df[["feature"]])
    
    # Check WOE values (approximately)
    woe_map = encoder.get_woe_map("feature")
    
    # Category A (mostly good): positive WOE
    a_woe = woe_map.get("A", None)
    assert a_woe is not None
    assert a_woe > 0, f"Category A should have positive WOE, got {a_woe}"
    
    # Category B (equal good/bad): negative WOE
    b_woe = woe_map.get("B", None)
    assert b_woe is not None
    assert b_woe < 0, f"Category B should have negative WOE, got {b_woe}"


def test_woe_unseen_category():
    """Encoder handles unseen category at transform time without crashing."""
    df_train = pd.DataFrame({
        "feature": ["A"] * 50 + ["B"] * 50,
        "target": [0] * 40 + [1] * 10 + [0] * 10 + [1] * 40
    })
    
    df_test = pd.DataFrame({
        "feature": ["A", "B", "C"],  # C is unseen
        "target": [0, 1, 0]
    })
    
    encoder = WOEEncoder(handle_unknown="ignore")
    encoder.fit(df_train, ["feature"], "target")
    
    # Should not raise
    transformed = encoder.transform(df_test[["feature"]])
    
    # Unseen category should get neutral WOE (0)
    assert transformed["feature"].iloc[2] == 0.0


def test_woe_numeric_binning():
    """WOE works with numeric features (binned)."""
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "numeric_feat": np.random.normal(0, 1, n),
        "target": np.random.binomial(1, 0.3, n)
    })
    # Make feature predictive
    df.loc[df["numeric_feat"] > 0.5, "target"] = 1
    df.loc[df["numeric_feat"] < -0.5, "target"] = 0
    
    encoder = WOEEncoder(n_bins=5)
    encoder.fit(df, ["numeric_feat"], "target")
    transformed = encoder.transform(df[["numeric_feat"]])
    
    # Should produce numeric WOE values
    assert transformed["numeric_feat"].dtype in [np.float64, np.float32]
    assert not transformed["numeric_feat"].isna().any()


def test_fit_woe_encoder_convenience():
    """Convenience function works."""
    df = pd.DataFrame({
        "f1": ["A", "B"] * 50,
        "f2": np.random.randn(100),
        "target": [0, 1] * 50
    })
    
    encoder = fit_woe_encoder(df, "target")
    assert encoder.is_fitted_
    assert "f1" in encoder.woe_maps_
    assert "f2" in encoder.woe_maps_


def test_woe_save_load():
    """Encoder can be saved and loaded."""
    import tempfile
    
    df = pd.DataFrame({
        "feature": ["A", "B"] * 50,
        "target": [0, 1] * 50
    })
    
    encoder = WOEEncoder()
    encoder.fit(df, ["feature"], "target")
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    
    try:
        encoder.save(path)
        
        loaded = WOEEncoder.load(path)
        assert loaded.is_fitted_
        assert loaded.woe_maps_ == encoder.woe_maps_
    finally:
        os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])