"""Credit decisioning: risk bands, threshold optimization, decision metrics."""


import numpy as np
import pandas as pd

from src.evaluate import evaluate_at_thresholds


def assign_risk_band(
    pd_scores: np.ndarray,
    low_threshold: float = 0.10,
    high_threshold: float = 0.30
) -> np.ndarray:
    """
    Assign risk bands based on PD thresholds.
    
    Returns: array of 'Low', 'Medium', 'High'
    """
    bands = np.full_like(pd_scores, "Medium", dtype=object)
    bands[pd_scores < low_threshold] = "Low"
    bands[pd_scores >= high_threshold] = "High"
    return bands


def decision_metrics(
    y_true: np.ndarray,
    pd_scores: np.ndarray,
    low_threshold: float = 0.10,
    high_threshold: float = 0.30
) -> dict[str, float]:
    """
    Calculate decision-layer metrics for a three-band policy.
    
    Low risk (PD < low_threshold) → Approve
    Medium risk → Review
    High risk (PD >= high_threshold) → Decline
    """
    bands = assign_risk_band(pd_scores, low_threshold, high_threshold)
    
    # Approve = Low risk
    approve_mask = bands == "Low"
    # Decline = High risk
    decline_mask = bands == "High"
    # Review = Medium risk
    review_mask = bands == "Medium"
    
    n_total = len(y_true)
    n_approve = approve_mask.sum()
    n_decline = decline_mask.sum()
    n_review = review_mask.sum()
    
    # Defaults among approved
    defaults_approved = y_true[approve_mask].sum() if n_approve > 0 else 0
    
    # Defaults among declined
    defaults_declined = y_true[decline_mask].sum() if n_decline > 0 else 0
    
    # Default capture rate: of all actual defaults, how many did we decline?
    total_defaults = y_true.sum()
    captured_defaults = defaults_declined
    capture_rate = captured_defaults / total_defaults if total_defaults > 0 else 0
    
    # Bad rate among approved
    bad_rate_approved = defaults_approved / n_approve if n_approve > 0 else 0
    
    # False decline rate: of all non-defaults, how many did we decline?
    total_non_defaults = (y_true == 0).sum()
    false_declines = ((y_true == 0) & decline_mask).sum()
    false_decline_rate = false_declines / total_non_defaults if total_non_defaults > 0 else 0
    
    return {
        "approval_rate": n_approve / n_total,
        "review_rate": n_review / n_total,
        "decline_rate": n_decline / n_total,
        "bad_rate_approved": bad_rate_approved,
        "default_capture_rate": capture_rate,
        "false_decline_rate": false_decline_rate,
        "n_approve": int(n_approve),
        "n_review": int(n_review),
        "n_decline": int(n_decline)
    }


def threshold_analysis(
    y_true: np.ndarray,
    pd_scores: np.ndarray,
    thresholds: list[float] | None = None
) -> pd.DataFrame:
    """
    Generate threshold analysis table showing approval rate vs risk trade-off.
    
    For each threshold (treat as approve if PD < threshold):
    - Approval rate
    - Bad rate among approved
    - Default capture rate
    - Expected loss (if EAD/LGD provided)
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.50, 50)
    
    return evaluate_at_thresholds(y_true, pd_scores, thresholds)


def find_optimal_threshold(
    y_true: np.ndarray,
    pd_scores: np.ndarray,
    objective: str = "min_bad_rate",
    min_approval: float = 0.0,
    max_bad_rate: float = 1.0,
    thresholds: list[float] | None = None
) -> dict:
    """
    Find threshold optimizing a business objective.
    
    Objectives:
    - "min_bad_rate": minimize bad rate among approved (subject to min_approval)
    - "max_approval": maximize approval rate (subject to max_bad_rate)
    - "balanced": maximize (approval_rate - bad_rate)
    """
    df = threshold_analysis(y_true, pd_scores, thresholds)
    
    # Apply constraints
    df = df[(df["approval_rate"] >= min_approval) & (df["bad_rate"] <= max_bad_rate)]
    
    if len(df) == 0:
        return {"threshold": None, "reason": "No threshold satisfies constraints"}
    
    if objective == "min_bad_rate":
        best = df.loc[df["bad_rate"].idxmin()]
    elif objective == "max_approval":
        best = df.loc[df["approval_rate"].idxmax()]
    elif objective == "balanced":
        df["score"] = df["approval_rate"] - df["bad_rate"]
        best = df.loc[df["score"].idxmax()]
    else:
        raise ValueError(f"Unknown objective: {objective}")
    
    return {
        "threshold": best["threshold"],
        "approval_rate": best["approval_rate"],
        "bad_rate": best["bad_rate"],
        "default_capture_rate": best["default_capture_rate"],
        "false_decline_rate": best["false_decline_rate"],
        "objective": objective
    }


def segment_analysis(
    y_true: np.ndarray,
    pd_scores: np.ndarray,
    segments: np.ndarray,
    low_threshold: float = 0.10,
    high_threshold: float = 0.30
) -> pd.DataFrame:
    """
    Analyze model performance across segments.
    
    Useful for fairness/monitoring analysis.
    """
    df = pd.DataFrame({
        "y_true": y_true,
        "pd": pd_scores,
        "segment": segments
    })
    
    rows = []
    for segment_name, group in df.groupby("segment"):
        if len(group) < 10:
            continue
        
        metrics = decision_metrics(
            group["y_true"].values,
            group["pd"].values,
            low_threshold,
            high_threshold
        )
        metrics["segment"] = segment_name
        metrics["sample_size"] = len(group)
        metrics["auc"] = 0.5
        try:
            from sklearn.metrics import roc_auc_score
            metrics["auc"] = roc_auc_score(group["y_true"], group["pd"])
        except Exception:
            pass
        
        rows.append(metrics)
    
    return pd.DataFrame(rows)