"""Model evaluation: AUC, PR-AUC, F1, Gini, KS, Brier, calibration, reliability curves."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    brier_score_loss, precision_recall_curve, roc_curve
)
from sklearn.model_selection import cross_val_predict, cross_validate
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Import predict_proba from models
from models import predict_proba


def gini_from_auc(auc: float) -> float:
    """Convert AUC to Gini coefficient."""
    return 2 * auc - 1


def ks_statistic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Kolmogorov-Smirnov statistic."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return max(tpr - fpr)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate all standard metrics."""
    return {
        "auc": roc_auc_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_pred),
        "gini": gini_from_auc(roc_auc_score(y_true, y_pred)),
        "ks": ks_statistic(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_pred),
        "f1": f1_score(y_true, (y_pred >= threshold).astype(int)),
    }


def reliability_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reliability curve (calibration curve).
    
    Returns:
    - bin_centers: mean predicted probability in each bin
    - bin_accuracies: observed default rate in each bin
    - bin_counts: number of samples in each bin
    """
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_pred, bins) - 1
    binids = np.clip(binids, 0, n_bins - 1)
    
    bin_sums = np.bincount(binids, weights=y_pred, minlength=n_bins)
    bin_true = np.bincount(binids, weights=y_true, minlength=n_bins)
    bin_counts = np.bincount(binids, minlength=n_bins)
    
    # Avoid division by zero
    mask = bin_counts > 0
    bin_centers = bin_sums[mask] / bin_counts[mask]
    bin_accuracies = bin_true[mask] / bin_counts[mask]
    bin_counts = bin_counts[mask]
    
    return bin_centers, bin_accuracies, bin_counts


def calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """Calculate calibration-specific metrics."""
    bin_centers, bin_accuracies, bin_counts = reliability_curve(y_true, y_pred, n_bins)
    
    # Expected Calibration Error (ECE)
    ece = np.average(np.abs(bin_centers - bin_accuracies), weights=bin_counts)
    
    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(bin_centers - bin_accuracies))
    
    # Calibration slope and intercept (logistic calibration)
    # Fit: logit(observed) = a + b * logit(predicted)
    eps = 1e-6
    logit_pred = np.log((y_pred + eps) / (1 - y_pred + eps))
    logit_obs = np.log((y_true + eps) / (1 - y_true + eps))
    
    # Use binned values for stability
    logit_bin_pred = np.log((bin_centers + eps) / (1 - bin_centers + eps))
    logit_bin_obs = np.log((bin_accuracies + eps) / (1 - bin_accuracies + eps))
    
    if len(bin_centers) > 1:
        slope, intercept, _, _, _ = stats.linregress(logit_bin_pred, logit_bin_obs)
    else:
        slope, intercept = 1.0, 0.0
    
    return {
        "ece": ece,
        "mce": mce,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "bin_centers": bin_centers.tolist(),
        "bin_accuracies": bin_accuracies.tolist(),
        "bin_counts": bin_counts.tolist()
    }


def evaluate_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    n_bins: int = 10
) -> Dict[str, any]:
    """Comprehensive evaluation including discrimination and calibration."""
    metrics = calculate_metrics(y_true, y_pred, threshold)
    cal_metrics = calibration_metrics(y_true, y_pred, n_bins)
    metrics.update(cal_metrics)
    return metrics


def evaluate_at_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: List[float]
) -> pd.DataFrame:
    """Evaluate metrics at multiple decision thresholds."""
    rows = []
    for thresh in thresholds:
        preds_binary = (y_pred >= thresh).astype(int)
        tp = ((preds_binary == 1) & (y_true == 1)).sum()
        fp = ((preds_binary == 1) & (y_true == 0)).sum()
        tn = ((preds_binary == 0) & (y_true == 0)).sum()
        fn = ((preds_binary == 0) & (y_true == 1)).sum()
        
        approval_rate = (preds_binary == 0).mean()  # Approve = low risk (PD < threshold)
        bad_rate = tp / (tp + fp) if (tp + fp) > 0 else 0
        capture_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_decline = fn / (fn + tn) if (fn + tn) > 0 else 0
        
        rows.append({
            "threshold": thresh,
            "approval_rate": approval_rate,
            "bad_rate": bad_rate,
            "default_capture_rate": capture_rate,
            "false_decline_rate": false_decline,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0
        })
    
    return pd.DataFrame(rows)


def compare_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """Compare multiple models on test set."""
    rows = []
    for name, preds in predictions.items():
        metrics = evaluate_all(y_true, preds)
        metrics["model"] = name
        rows.append(metrics)
    return pd.DataFrame(rows)


def print_calibration_table(metrics: Dict) -> None:
    """Print calibration table: Predicted PD bucket vs Observed default rate."""
    centers = metrics.get("bin_centers", [])
    accuracies = metrics.get("bin_accuracies", [])
    counts = metrics.get("bin_counts", [])
    
    if not centers:
        return
    
    print("\nCalibration Table:")
    print(f"{'Predicted PD Bucket':<25} {'Observed Default Rate':<25} {'Count':>10}")
    print("-" * 60)
    for c, a, n in zip(centers, accuracies, counts):
        print(f"{c:.2%}{'':<15} {a:.2%}{'':<15} {n:>10}")


def evaluate_cv(
    model_builder: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Evaluate model using cross-validation.
    
    Uses cross_val_predict to get out-of-fold predictions, then evaluates.
    """
    # Get out-of-fold predictions
    # Need to wrap model_builder to work with cross_val_predict
    from sklearn.base import BaseEstimator
    from sklearn.pipeline import Pipeline
    
    # Build a fresh model for CV
    model = model_builder(X, y)
    
    # Get out-of-fold predictions
    try:
        y_pred_oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    except Exception:
        # Fallback: train on each fold manually
        y_pred_oof = np.zeros(len(y))
        for train_idx, val_idx in cv.split(X, y):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            fold_model = model_builder(X_fold_train, y_fold_train)
            y_pred_oof[val_idx] = predict_proba(fold_model, X_fold_val)
    
    # Evaluate
    metrics = evaluate_all(y.values, y_pred_oof, n_bins=n_bins)
    
    # Also compute per-fold AUC for std
    from sklearn.model_selection import cross_validate
    try:
        cv_results = cross_validate(model, X, y, cv=cv, scoring="roc_auc", return_train_score=False)
        auc_mean = cv_results["test_score"].mean()
        auc_std = cv_results["test_score"].std()
    except Exception:
        # Fallback: use overall AUC
        auc_mean = metrics["auc"]
        auc_std = 0.0
    
    return {
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "brier_mean": metrics["brier"],
        "brier_std": 0.0,  # Could compute per-fold if needed
        **metrics
    }