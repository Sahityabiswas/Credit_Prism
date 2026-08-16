"""SHAP explainability wrapper for global and local explanations."""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def generate_shap_explanations(
    model: Any,
    X: pd.DataFrame,
    sample_size: Optional[int] = None
) -> shap.Explanation:
    """
    Generate SHAP values for model explanations.
    
    Uses TreeExplainer for tree models, LinearExplainer for linear,
    falls back to KernelExplainer (slower).
    """
    # Sample for speed if needed
    if sample_size and len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X
    
    # Try to use appropriate explainer
    try:
        # Check for tree-based models (RF, XGB)
        if hasattr(model, "estimators_") or hasattr(model, "base_estimator_"):
            # Get base estimator if calibrated
            base = getattr(model, "base_estimator_", model)
            if hasattr(base, "estimators_"):  # RF
                explainer = shap.TreeExplainer(base)
            elif hasattr(base, "feature_importances_"):  # XGB
                explainer = shap.TreeExplainer(base)
            else:
                raise ValueError("Unknown tree model")
        elif hasattr(model, "coef_"):  # Linear
            explainer = shap.LinearExplainer(model, X_sample)
        else:
            # Fallback to kernel explainer
            explainer = shap.KernelExplainer(model.predict_proba, X_sample)
    except Exception as e:
        print(f"Using KernelExplainer as fallback: {e}")
        explainer = shap.KernelExplainer(model.predict_proba, X_sample)
    
    shap_values = explainer(X_sample)
    return shap_values


def save_global_shap_plot(
    shap_values: shap.Explanation,
    output_path: str = "artifacts/shap_plots/shap_summary.png",
    max_display: int = 20
) -> None:
    """Save global SHAP summary plot (beeswarm), fall back to bar plot."""
    plt.figure(figsize=(10, 8))
    try:
        shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    except Exception:
        # Fallback: simple mean |SHAP| bar plot
        try:
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
            feature_names = shap_values.feature_names if hasattr(shap_values, 'feature_names') else [f"f{i}" for i in range(len(mean_abs_shap))]
            # Sort
            idx = np.argsort(mean_abs_shap)[::-1][:max_display]
            plt.barh(range(len(idx)), mean_abs_shap[idx])
            plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
            plt.gca().invert_yaxis()
            plt.xlabel("Mean |SHAP|")
        except Exception:
            plt.text(0.5, 0.5, "SHAP plot not available for this model type", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_shap_bar_plot(
    shap_values: shap.Explanation,
    output_path: str = "artifacts/shap_plots/shap_bar.png",
    max_display: int = 20
) -> None:
    """Save global SHAP bar plot (mean |SHAP|)."""
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def pick_diverse_examples(
    shap_values: shap.Explanation,
    n: int = 3,
    y_pred: Optional[np.ndarray] = None
) -> List[int]:
    """
    Pick diverse examples for local explanations.
    
    Strategy: one low-risk, one high-risk, one borderline (if y_pred provided),
    otherwise pick by SHAP magnitude diversity.
    """
    if y_pred is not None:
        # Sort by predicted probability
        sorted_idx = np.argsort(y_pred)
        n_low = n // 3
        n_high = n // 3
        n_mid = n - n_low - n_high
        
        low_idx = sorted_idx[:n_low]
        high_idx = sorted_idx[-n_high:]
        mid_start = len(sorted_idx) // 2 - n_mid // 2
        mid_idx = sorted_idx[mid_start:mid_start + n_mid]
        
        return list(low_idx) + list(mid_idx) + list(high_idx)
    
    # Fallback: pick by SHAP value spread
    mean_abs_shap = np.abs(shap_values.values).mean(axis=1)
    sorted_idx = np.argsort(mean_abs_shap)
    
    indices = []
    step = len(sorted_idx) // n
    for i in range(n):
        idx = min(i * step + step // 2, len(sorted_idx) - 1)
        indices.append(sorted_idx[idx])
    
    return indices


def save_local_shap_examples(
    shap_values: shap.Explanation,
    output_dir: str = "artifacts/shap_plots",
    n: int = 3,
    y_pred: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None
) -> List[str]:
    """
    Save local SHAP waterfall plots for diverse examples.
    
    Returns list of saved file paths.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    indices = pick_diverse_examples(shap_values, n, y_pred)
    saved_paths = []
    
    for i, idx in enumerate(indices):
        plt.figure(figsize=(10, 6))
        
        try:
            # Create single-row explanation
            single_explanation = shap_values[idx:idx+1]
            
            # Waterfall plot
            shap.plots.waterfall(single_explanation[0], max_display=15, show=False)
            
            # Determine risk level for filename
            if y_pred is not None:
                pd_val = y_pred[idx]
                if pd_val < 0.1:
                    risk_label = "low_risk"
                elif pd_val > 0.3:
                    risk_label = "high_risk"
                else:
                    risk_label = "borderline"
            else:
                risk_label = f"example_{i}"
            
            filepath = f"{output_dir}/applicant_{risk_label}_{i}.png"
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            saved_paths.append(filepath)
        except Exception as e:
            print(f"  Warning: Could not generate local SHAP plot for index {idx}: {e}")
            plt.close()
            continue
        finally:
            plt.close()
    
    return saved_paths


def compare_shap_with_logreg(
    shap_values: shap.Explanation,
    logreg_model: Any,
    feature_names: List[str],
    top_k: int = 10
) -> pd.DataFrame:
    """
    Compare top SHAP drivers with Logistic Regression coefficients.
    
    Returns DataFrame with both rankings for auditability check.
    """
    # Mean absolute SHAP per feature
    shap_vals = shap_values.values
    # Handle multi-dimensional SHAP values (e.g., for multi-output models)
    if shap_vals.ndim > 2:
        shap_vals = shap_vals.reshape(shap_vals.shape[0], -1)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    
    # Ensure feature_names matches
    n_features = len(mean_abs_shap)
    if len(feature_names) != n_features:
        feature_names = feature_names[:n_features] if len(feature_names) > n_features else feature_names + [f"f{i}" for i in range(len(feature_names), n_features)]
    
    shap_ranking = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).head(top_k)
    
    # LogReg coefficients (absolute)
    if hasattr(logreg_model, "coef_"):
        coef = np.abs(logreg_model.coef_[0])
    elif hasattr(logreg_model, "base_estimator_") and hasattr(logreg_model.base_estimator_, "coef_"):
        coef = np.abs(logreg_model.base_estimator_.coef_[0])
    else:
        return shap_ranking
    
    logreg_ranking = pd.DataFrame({
        "feature": feature_names,
        "abs_coef": coef
    }).sort_values("abs_coef", ascending=False).head(top_k)
    
    # Merge
    merged = shap_ranking.merge(logreg_ranking, on="feature", how="outer")
    merged["shap_rank"] = range(1, len(merged) + 1)
    merged["logreg_rank"] = merged["abs_coef"].rank(ascending=False).astype(int)
    merged["rank_diff"] = (merged["shap_rank"] - merged["logreg_rank"]).abs()
    
    return merged.sort_values("shap_rank")


def shap_dependence_plots(
    shap_values: shap.Explanation,
    features: List[str],
    output_dir: str = "artifacts/shap_plots",
    interaction_index: str = "auto"
) -> List[str]:
    """Generate SHAP dependence plots for top features."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    saved = []
    for feat in features:
        plt.figure(figsize=(8, 6))
        shap.plots.scatter(
            shap_values[:, feat],
            color=shap_values,
            show=False
        )
        filepath = f"{output_dir}/shap_dependence_{feat}.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(filepath)
    
    return saved