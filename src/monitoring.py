"""Population Stability Index (PSI) and drift monitoring."""

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def calculate_psi(
    expected_scores: np.ndarray,
    actual_scores: np.ndarray,
    buckets: int = 10
) -> float:
    """
    Calculate Population Stability Index (PSI).
    
    PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
    
    Interpretation:
    - PSI < 0.10: stable
    - 0.10 - 0.25: moderate shift (investigate)
    - > 0.25: significant drift (trigger review)
    """
    # Use quantile bins based on expected (reference) distribution
    expected = np.asarray(expected_scores)
    actual = np.asarray(actual_scores)
    
    # Remove NaN
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    
    # Create bins from expected quantiles
    try:
        bin_edges = np.quantile(expected, np.linspace(0, 1, buckets + 1))
        # Ensure unique edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0
    except Exception:
        # Fallback to equal-width bins
        bin_edges = np.linspace(expected.min(), expected.max(), buckets + 1)
    
    # Bin both distributions
    expected_bins = np.digitize(expected, bin_edges) - 1
    actual_bins = np.digitize(actual, bin_edges) - 1
    
    # Clip to valid range
    n_bins = len(bin_edges) - 1
    expected_bins = np.clip(expected_bins, 0, n_bins - 1)
    actual_bins = np.clip(actual_bins, 0, n_bins - 1)
    
    # Calculate percentages
    expected_counts = np.bincount(expected_bins, minlength=n_bins)
    actual_counts = np.bincount(actual_bins, minlength=n_bins)
    
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # Avoid division by zero
    eps = 1e-10
    expected_pct = np.maximum(expected_pct, eps)
    actual_pct = np.maximum(actual_pct, eps)
    
    # PSI contribution per bucket
    psi_contrib = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    
    return float(np.sum(psi_contrib))


def psi_interpretation(psi: float) -> str:
    """Interpret PSI value."""
    if psi < 0.10:
        return "stable"
    elif psi < 0.25:
        return "moderate_shift"
    else:
        return "significant_drift"


def calculate_psi_detailed(
    expected_scores: np.ndarray,
    actual_scores: np.ndarray,
    buckets: int = 10
) -> dict:
    """Calculate PSI with detailed bucket breakdown."""
    expected = np.asarray(expected_scores)
    actual = np.asarray(actual_scores)
    
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    bin_edges = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    bin_edges = np.unique(bin_edges)
    n_bins = len(bin_edges) - 1
    
    expected_bins = np.digitize(expected, bin_edges) - 1
    actual_bins = np.digitize(actual, bin_edges) - 1
    
    expected_bins = np.clip(expected_bins, 0, n_bins - 1)
    actual_bins = np.clip(actual_bins, 0, n_bins - 1)
    
    expected_counts = np.bincount(expected_bins, minlength=n_bins)
    actual_counts = np.bincount(actual_bins, minlength=n_bins)
    
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    eps = 1e-10
    expected_pct = np.maximum(expected_pct, eps)
    actual_pct = np.maximum(actual_pct, eps)
    
    psi_contrib = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = float(np.sum(psi_contrib))
    
    buckets_detail = []
    for i in range(n_bins):
        buckets_detail.append({
            "bucket": i,
            "bin_edge_lower": float(bin_edges[i]),
            "bin_edge_upper": float(bin_edges[i + 1]),
            "expected_count": int(expected_counts[i]),
            "actual_count": int(actual_counts[i]),
            "expected_pct": float(expected_pct[i]),
            "actual_pct": float(actual_pct[i]),
            "psi_contribution": float(psi_contrib[i])
        })
    
    return {
        "psi": psi,
        "interpretation": psi_interpretation(psi),
        "buckets": buckets_detail,
        "n_expected": len(expected),
        "n_actual": len(actual)
    }


def calculate_variable_psi(
    train_df: pd.DataFrame,
    oot_df: pd.DataFrame,
    feature_cols: list[str],
    buckets: int = 10
) -> pd.DataFrame:
    """Calculate PSI for multiple variables (features and scores)."""
    rows = []
    
    for col in feature_cols:
        if col not in train_df.columns or col not in oot_df.columns:
            continue
        
        train_vals = train_df[col].values
        oot_vals = oot_df[col].values
        
        # Only calculate for numeric features
        if pd.api.types.is_numeric_dtype(train_df[col]):
            psi_result = calculate_psi_detailed(train_vals, oot_vals, buckets)
            rows.append({
                "variable": col,
                "psi": psi_result["psi"],
                "interpretation": psi_result["interpretation"]
            })
    
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def generate_psi_report(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    oot_scores: np.ndarray,
    train_df: pd.DataFrame = None,
    oot_df: pd.DataFrame = None,
    feature_cols: list[str] = None,
    buckets: int = 10
) -> dict:
    """Generate comprehensive PSI report."""
    report = {
        "score_psi_test_vs_train": calculate_psi_detailed(train_scores, test_scores, buckets),
        "score_psi_oot_vs_train": calculate_psi_detailed(train_scores, oot_scores, buckets),
        "score_psi_oot_vs_test": calculate_psi_detailed(test_scores, oot_scores, buckets)
    }
    
    # Add variable-level PSI if data provided
    if train_df is not None and oot_df is not None and feature_cols:
        report["variable_psi"] = calculate_variable_psi(
            train_df, oot_df, feature_cols, buckets
        ).to_dict("records")
    
    return report


def save_psi_report(report: dict, path: str) -> None:
    """Save PSI report to JSON."""
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)


def load_psi_report(path: str) -> dict:
    """Load PSI report from JSON."""
    with open(path) as f:
        return json.load(f)