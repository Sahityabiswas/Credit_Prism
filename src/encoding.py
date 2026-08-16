"""WOE (Weight of Evidence) Encoder — built from scratch for transparency."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import json


class WOEEncoder:
    """
    Weight of Evidence encoder for credit risk modeling.
    
    WOE = ln(good_pct / bad_pct) for each bin/category.
    Positive WOE = lower risk (more goods), Negative WOE = higher risk (more bads).
    """
    
    def __init__(
        self,
        n_bins: int = 10,
        min_bin_size: float = 0.05,
        handle_unknown: str = "ignore",
        random_state: int = 42
    ):
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self.handle_unknown = handle_unknown  # "ignore" or "error"
        self.random_state = random_state
        
        self.woe_maps_: Dict[str, Dict] = {}
        self.bin_edges_: Dict[str, np.ndarray] = {}
        self.feature_types_: Dict[str, str] = {}  # "numeric" or "categorical"
        self.global_good_: int = 0
        self.global_bad_: int = 0
        self.is_fitted_ = False
    
    def _bin_numeric(self, series: pd.Series, target: pd.Series) -> np.ndarray:
        """Create equal-frequency bins for numeric feature."""
        # Use quantile-based binning
        try:
            # Try quantile binning
            _, bins = pd.qcut(
                series,
                q=self.n_bins,
                duplicates="drop",
                retbins=True
            )
        except ValueError:
            # Fallback: equal-width bins
            _, bins = pd.cut(series, bins=self.n_bins, retbins=True)
        
        # Ensure min_bin_size constraint by merging small bins
        bins = self._enforce_min_bin_size(series, target, bins)
        return bins
    
    def _enforce_min_bin_size(
        self,
        series: pd.Series,
        target: pd.Series,
        bins: np.ndarray
    ) -> np.ndarray:
        """Merge adjacent bins that are too small."""
        min_count = len(series) * self.min_bin_size
        bin_labels = pd.cut(series, bins=bins, include_lowest=True)
        bin_counts = bin_labels.value_counts().sort_index()
        
        # Simple approach: if any bin is too small, reduce n_bins
        while (bin_counts < min_count).any() and len(bins) > 3:
            # Re-bin with fewer bins
            try:
                _, bins = pd.qcut(
                    series,
                    q=len(bins) - 2,
                    duplicates="drop",
                    retbins=True
                )
                bin_labels = pd.cut(series, bins=bins, include_lowest=True)
                bin_counts = bin_labels.value_counts().sort_index()
            except ValueError:
                break
        
        return bins
    
    def _compute_woe(self, good_count: int, bad_count: int) -> float:
        """Compute WOE for a single bin."""
        # Add small smoothing to avoid division by zero
        eps = 1e-6
        good_pct = (good_count + eps) / (self.global_good_ + eps)
        bad_pct = (bad_count + eps) / (self.global_bad_ + eps)
        return np.log(good_pct / bad_pct)
    
    def fit(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> "WOEEncoder":
        """
        Fit WOE encoder on training data.
        
        Parameters:
        - df: training DataFrame
        - feature_cols: list of feature column names to encode
        - target_col: target column name (binary 0/1)
        """
        self.global_good_ = (df[target_col] == 0).sum()
        self.global_bad_ = (df[target_col] == 1).sum()
        
        if self.global_good_ == 0 or self.global_bad_ == 0:
            raise ValueError("Target must contain both classes (0 and 1)")
        
        for feature in feature_cols:
            if feature not in df.columns:
                continue
            
            series = df[feature]
            target = df[target_col]
            
            # Determine feature type
            if pd.api.types.is_numeric_dtype(series):
                self.feature_types_[feature] = "numeric"
                bins = self._bin_numeric(series, target)
                self.bin_edges_[feature] = bins
                binned = pd.cut(series, bins=bins, include_lowest=True)
            else:
                self.feature_types_[feature] = "categorical"
                binned = series.astype(str)  # Treat as categories
            
            # Compute WOE per bin
            woe_map = {}
            cross = pd.DataFrame({"bin": binned, "target": target})
            
            for bin_val, group in cross.groupby("bin", observed=False):
                good = (group["target"] == 0).sum()
                bad = (group["target"] == 1).sum()
                woe = self._compute_woe(good, bad)
                woe_map[str(bin_val)] = woe
            
            # Handle unseen categories at transform time
            woe_map["__UNSEEN__"] = 0.0  # Neutral WOE
            
            self.woe_maps_[feature] = woe_map
        
        self.is_fitted_ = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features to WOE values."""
        if not self.is_fitted_:
            raise ValueError("Encoder must be fitted before transform")
        
        df = df.copy()
        
        for feature, woe_map in self.woe_maps_.items():
            if feature not in df.columns:
                continue
            
            series = df[feature]
            ftype = self.feature_types_.get(feature, "categorical")
            
            if ftype == "numeric":
                bins = self.bin_edges_[feature]
                binned = pd.cut(series, bins=bins, include_lowest=True)
                # Convert to string for mapping
                binned_str = binned.astype(str)
            else:
                binned_str = series.astype(str)
            
            # Map to WOE, handle unseen
            def map_woe(val):
                if val in woe_map:
                    return woe_map[val]
                elif self.handle_unknown == "error":
                    raise ValueError(f"Unseen value '{val}' in feature '{feature}'")
                else:
                    return woe_map["__UNSEEN__"]
            
            df[feature] = binned_str.apply(map_woe)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, feature_cols, target_col).transform(df)
    
    def get_woe_map(self, feature: str) -> Dict:
        """Get WOE mapping for a feature (for inspection)."""
        return self.woe_maps_.get(feature, {})
    
    def save(self, path: str) -> None:
        """Save encoder to JSON."""
        data = {
            "n_bins": self.n_bins,
            "min_bin_size": self.min_bin_size,
            "handle_unknown": self.handle_unknown,
            "random_state": self.random_state,
            "woe_maps": {k: {str(kk): vv for kk, vv in v.items()} for k, v in self.woe_maps_.items()},
            "bin_edges": {k: v.tolist() for k, v in self.bin_edges_.items()},
            "feature_types": self.feature_types_,
            "global_good": int(self.global_good_),
            "global_bad": int(self.global_bad_),
            "is_fitted": self.is_fitted_
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "WOEEncoder":
        """Load encoder from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        
        encoder = cls(
            n_bins=data["n_bins"],
            min_bin_size=data["min_bin_size"],
            handle_unknown=data["handle_unknown"],
            random_state=data["random_state"]
        )
        encoder.woe_maps_ = data["woe_maps"]
        encoder.bin_edges_ = {k: np.array(v) for k, v in data["bin_edges"].items()}
        encoder.feature_types_ = data["feature_types"]
        encoder.global_good_ = data["global_good"]
        encoder.global_bad_ = data["global_bad"]
        encoder.is_fitted_ = data["is_fitted"]
        return encoder


def fit_woe_encoder(
    df: pd.DataFrame,
    target: str,
    feature_cols: Optional[List[str]] = None,
    n_bins: int = 10,
    min_bin_size: float = 0.05
) -> WOEEncoder:
    """Convenience function to fit WOE encoder."""
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target]
    
    encoder = WOEEncoder(n_bins=n_bins, min_bin_size=min_bin_size)
    encoder.fit(df, feature_cols, target)
    return encoder