"""Expected Loss calculation: PD × EAD × LGD."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union


def expected_loss(
    pd: Union[float, np.ndarray, pd.Series],
    ead: Union[float, np.ndarray, pd.Series],
    lgd: Union[float, np.ndarray, pd.Series]
) -> Union[float, np.ndarray]:
    """
    Calculate Expected Loss = PD × EAD × LGD.
    
    Parameters:
    - pd: Probability of Default (0-1)
    - ead: Exposure at Default (currency amount)
    - lgd: Loss Given Default (0-1)
    
    Returns: Expected Loss in same currency as EAD
    """
    return pd * ead * lgd


def portfolio_expected_loss(
    df: pd.DataFrame,
    pd_col: str = "pd",
    ead_col: str = "ead",
    lgd_col: str = "lgd"
) -> float:
    """Calculate total expected loss for a portfolio."""
    return (df[pd_col] * df[ead_col] * df[lgd_col]).sum()


def expected_loss_by_band(
    df: pd.DataFrame,
    pd_col: str = "pd",
    ead_col: str = "ead",
    lgd_col: str = "lgd",
    band_col: str = "risk_band"
) -> pd.DataFrame:
    """Break down expected loss by risk band."""
    df = df.copy()
    df["el"] = expected_loss(df[pd_col], df[ead_col], df[lgd_col])
    
    summary = df.groupby(band_col).agg(
        n_accounts=("el", "count"),
        total_ead=(ead_col, "sum"),
        total_el=("el", "sum"),
        avg_pd=(pd_col, "mean"),
        avg_lgd=(lgd_col, "mean")
    ).reset_index()
    
    summary["el_pct_of_ead"] = summary["total_el"] / summary["total_ead"]
    return summary


def sensitivity_analysis(
    pd: np.ndarray,
    ead_range: tuple = (5000, 50000),
    lgd_range: tuple = (0.2, 0.6),
    n_points: int = 5
) -> pd.DataFrame:
    """
    Perform sensitivity analysis over EAD/LGD assumptions.
    
    Returns DataFrame with expected loss across assumption grid.
    """
    ead_values = np.linspace(ead_range[0], ead_range[1], n_points)
    lgd_values = np.linspace(lgd_range[0], lgd_range[1], n_points)
    
    rows = []
    for ead in ead_values:
        for lgd in lgd_values:
            el = expected_loss(pd, ead, lgd)
            rows.append({
                "ead": ead,
                "lgd": lgd,
                "mean_el": el.mean(),
                "total_el": el.sum(),
                "median_el": np.median(el)
            })
    
    return pd.DataFrame(rows)


def el_with_assumptions(
    pd_scores: np.ndarray,
    ead_default: float = 10000,
    lgd_default: float = 0.45
) -> Dict:
    """
    Calculate EL with explicit assumptions.
    
    Returns dict with results and assumption documentation.
    """
    total_el = expected_loss(pd_scores, ead_default, lgd_default).sum()
    mean_el = expected_loss(pd_scores, ead_default, lgd_default).mean()
    
    return {
        "total_expected_loss": total_el,
        "mean_expected_loss_per_account": mean_el,
        "assumptions": {
            "ead": ead_default,
            "lgd": lgd_default,
            "note": "EAD/LGD are assumed constants. Actual portfolio EL requires account-level EAD and LGD data."
        }
    }


def el_by_decision(
    df: pd.DataFrame,
    pd_col: str = "pd",
    ead_col: str = "ead",
    lgd_col: str = "lgd",
    decision_col: str = "decision"
) -> pd.DataFrame:
    """Expected loss broken down by decision (Approve/Review/Decline)."""
    df = df.copy()
    df["el"] = expected_loss(df[pd_col], df[ead_col], df[lgd_col])
    
    summary = df.groupby(decision_col).agg(
        n_accounts=("el", "count"),
        total_ead=(ead_col, "sum"),
        total_el=("el", "sum"),
        avg_pd=(pd_col, "mean")
    ).reset_index()
    
    # EL if we approve all vs current policy
    total_el_if_approve_all = df["el"].sum()
    summary["el_saved_vs_approve_all"] = total_el_if_approve_all - summary["total_el"]
    
    return summary