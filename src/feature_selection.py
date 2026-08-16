"""Information Value (IV) calculation and feature selection utilities."""


import numpy as np
import pandas as pd


def calculate_iv(
    df: pd.DataFrame,
    target: str | pd.Series,
    feature_cols: list[str] | None = None
) -> pd.DataFrame:
    """
    Calculate Information Value (IV) for each feature.
    
    IV = sum((good_pct - bad_pct) * ln(good_pct / bad_pct)) across bins
    
    IV interpretation:
    - < 0.02: not useful
    - 0.02 - 0.1: weak
    - 0.1 - 0.3: medium
    - 0.3 - 0.5: strong
    - > 0.5: suspicious (potential leakage)
    """
    # Handle target as column name or Series
    if isinstance(target, pd.Series):
        target_series = target
        target_name = target.name or "target"
    else:
        target_name = target
        target_series = df[target]
    
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_name]
    
    total_good = (target_series == 0).sum()
    total_bad = (target_series == 1).sum()
    
    if total_good == 0 or total_bad == 0:
        raise ValueError("Target must contain both classes")
    
    eps = 1e-10
    results = []
    
    for feature in feature_cols:
        if feature not in df.columns:
            continue
        
        df[feature]
        
        # Group by feature values (already binned/WOE-transformed or categorical)
        grouped = df.groupby(feature, observed=False)[target_name].agg(
            good=lambda x: (x == 0).sum(),
            bad=lambda x: (x == 1).sum()
        ).reset_index()
        
        grouped["good_pct"] = (grouped["good"] + eps) / (total_good + eps)
        grouped["bad_pct"] = (grouped["bad"] + eps) / (total_bad + eps)
        
        # IV contribution per bin
        grouped["iv_contrib"] = (
            (grouped["good_pct"] - grouped["bad_pct"]) *
            np.log(grouped["good_pct"] / grouped["bad_pct"])
        )
        
        feature_iv = grouped["iv_contrib"].sum()
        n_bins = len(grouped)
        
        results.append({
            "feature": feature,
            "iv": feature_iv,
            "n_bins": n_bins
        })
    
    iv_df = pd.DataFrame(results).sort_values("iv", ascending=False).reset_index(drop=True)
    return iv_df


def select_by_iv(
    iv_df: pd.DataFrame,
    iv_min: float = 0.02,
    iv_max: float = 0.5
) -> list[str]:
    """Select features within IV range."""
    selected = iv_df[
        (iv_df["iv"] >= iv_min) & (iv_df["iv"] <= iv_max)
    ]["feature"].tolist()
    return selected


def calculate_vif(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Calculate Variance Inflation Factor for each feature."""
    from sklearn.linear_model import LinearRegression
    
    vif_data = []
    for feature in feature_cols:
        other_features = [f for f in feature_cols if f != feature]
        if not other_features:
            vif_data.append({"feature": feature, "vif": 1.0})
            continue
        
        X = df[other_features]
        y = df[feature]
        
        # Handle constant features
        if y.nunique() <= 1:
            vif_data.append({"feature": feature, "vif": np.inf})
            continue
        
        reg = LinearRegression().fit(X, y)
        r2 = reg.score(X, y)
        
        vif = np.inf if r2 >= 1.0 else 1 / (1 - r2)
        
        vif_data.append({"feature": feature, "vif": vif})
    
    return pd.DataFrame(vif_data)


def prune_by_correlation(
    df: pd.DataFrame,
    threshold: float = 0.8,
    method: str = "pearson"
) -> list[str]:
    """
    Remove highly correlated features, keeping the one with higher IV (if available).
    
    Returns list of selected feature names.
    """
    corr_matrix = df.corr(method=method).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > threshold].tolist()
        if correlated:
            # Keep first (assumed higher IV if pre-sorted), drop rest
            for c in correlated:
                if c not in to_drop:
                    to_drop.add(col)
                    break
    
    selected = [c for c in df.columns if c not in to_drop]
    return selected


def prune_by_vif(
    df: pd.DataFrame,
    feature_cols: list[str],
    vif_threshold: float = 10.0,
    max_iter: int = 10
) -> list[str]:
    """Iteratively remove features with highest VIF until all below threshold."""
    remaining = feature_cols.copy()
    
    for _ in range(max_iter):
        vif_df = calculate_vif(df, remaining)
        max_vif_row = vif_df.loc[vif_df["vif"].idxmax()]
        
        if max_vif_row["vif"] <= vif_threshold:
            break
        
        # Remove feature with highest VIF
        remaining.remove(max_vif_row["feature"])
        print(f"Dropped {max_vif_row['feature']} (VIF={max_vif_row['vif']:.2f})")
    
    return remaining


def select_features(
    df: pd.DataFrame,
    target: str,
    iv_min: float = 0.02,
    iv_max: float = 0.5,
    correlation_threshold: float = 0.8,
    vif_threshold: float = 10.0
) -> tuple[list[str], pd.DataFrame]:
    """
    Full feature selection pipeline:
    1. Calculate IV
    2. Filter by IV range
    3. Prune by correlation
    4. Prune by VIF
    
    Returns (selected_features, iv_table)
    """
    # Step 1: IV calculation
    iv_table = calculate_iv(df, target)
    print(f"IV calculated for {len(iv_table)} features")
    print(iv_table.head(10).to_string())
    
    # Step 2: IV filtering
    selected = select_by_iv(iv_table, iv_min, iv_max)
    print(f"After IV filter ({iv_min}-{iv_max}): {len(selected)} features")
    
    if not selected:
        return [], iv_table
    
    # Step 3: Correlation pruning
    selected = prune_by_correlation(df[selected], correlation_threshold)
    print(f"After correlation pruning (>{correlation_threshold}): {len(selected)} features")
    
    # Step 4: VIF pruning
    selected = prune_by_vif(df, selected, vif_threshold)
    print(f"After VIF pruning (<{vif_threshold}): {len(selected)} features")
    
    return selected, iv_table