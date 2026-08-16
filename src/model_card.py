"""Automated model card generation from pipeline outputs."""

import json
from datetime import datetime

import pandas as pd


def generate_model_card(
    model_name: str,
    config: dict,
    metrics_report: dict,
    oot_metrics: dict,
    psi_score: float,
    iv_table: pd.DataFrame,
    selected_features: list[str],
    shap_comparison: pd.DataFrame | None = None,
    decision_metrics: dict | None = None,
    expected_loss_info: dict | None = None,
    limitations: list[str] | None = None,
    cv_results: dict | None = None
) -> str:
    """Generate model card as Markdown string."""
    
    lines = []
    
    # Header
    lines.append(f"# Model Card: {model_name}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Business Objective
    lines.append("## 1. Business Objective")
    lines.append("")
    lines.append("Credit risk probability-of-default (PD) modeling for lending decisions.")
    lines.append("The model estimates the probability that an applicant will default,")
    lines.append("enabling risk-based pricing, approval/decline decisions, and portfolio monitoring.")
    lines.append("")
    
    # Target Definition
    lines.append("## 2. Target Definition")
    lines.append("")
    lines.append(f"- **Target:** {config.get('target', 'default')}")
    lines.append("- **Definition:** Binary indicator (1 = default, 0 = non-default)")
    lines.append("- **Observation window:** Defined by data source")
    lines.append("")
    
    # Data
    lines.append("## 3. Data")
    lines.append("")
    lines.append("- **Dataset:** Home Credit Default Risk (application_train)")
    lines.append(f"- **Rows:** {config.get('raw_path', 'N/A')}")
    lines.append(f"- **Target:** {config.get('target', 'TARGET')} (binary, ~8% default rate)")
    lines.append(f"- **Split strategy:** Stratified random split (test_size={config.get('test_size', 0.2)}) + {config.get('cv_folds', 5)}-fold CV")
    lines.append("- **Critical limitation:** No true temporal ordering exists in Home Credit data. All temporal fields (DAYS_BIRTH, DAYS_EMPLOYED, etc.) are relative to the current application date. No calendar dates or true application timestamps exist. Therefore, true chronological/out-of-time validation is NOT possible.")
    lines.append(f"- **Leakage checks:** {len(config.get('forbidden_columns', []))} forbidden columns excluded")
    lines.append("")
    
    # Features
    lines.append("## 4. Features")
    lines.append("")
    lines.append(f"- **Total features evaluated:** {len(iv_table)}")
    lines.append(f"- **Features selected:** {len(selected_features)}")
    lines.append(f"- **Selection criteria:** IV in [{config.get('iv_min', 0.02)}, {config.get('iv_max', 2.0)}], correlation < {config.get('correlation_threshold', 0.8)}, VIF < {config.get('vif_threshold', 10)}")
    lines.append("")
    lines.append("### Selected Features")
    lines.append("")
    for feat in selected_features:
        iv_row = iv_table[iv_table["feature"] == feat]
        iv_val = iv_row["iv"].values[0] if len(iv_row) > 0 else "N/A"
        lines.append(f"- {feat} (IV: {iv_val:.4f})")
    lines.append("")
    
    # Model
    lines.append("## 5. Model")
    lines.append("")
    lines.append(f"- **Algorithm:** {model_name}")
    lines.append("- **Framework:** scikit-learn / XGBoost")
    lines.append("- **Calibration:** Isotonic regression (3-fold CV)")
    lines.append("- **Class weighting:** Balanced")
    lines.append("")
    
    # Test Metrics
    lines.append("## 6. Evaluation Metrics (Holdout Test)")
    lines.append("")
    test_metrics = metrics_report.get(model_name, {})
    if test_metrics:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key, val in test_metrics.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                lines.append(f"| {key} | {val:.4f} |")
        lines.append("")
    
    # Cross-Validation / Validation Results
    if cv_results:
        lines.append("## 7. Validation Results")
        lines.append("")
        
        # Check format - quick validation has 'auc', 'brier', 'ece' keys
        first_key = list(cv_results.keys())[0]
        first_val = cv_results[first_key]
        
        if 'auc_mean' in first_val:
            # Full CV format
            lines.append("| Model | CV AUC (mean) | CV AUC (std) | CV Brier (mean) |")
            lines.append("|--------|---------------|--------------|-----------------|")
            for model_key, cv_metrics in cv_results.items():
                auc_mean = cv_metrics.get('auc_mean', 'N/A')
                auc_std = cv_metrics.get('auc_std', 'N/A')
                brier_mean = cv_metrics.get('brier_mean', 'N/A')
                if isinstance(auc_mean, float):
                    lines.append(f"| {model_key} | {auc_mean:.4f} | {auc_std:.4f} | {brier_mean:.4f} |")
                else:
                    lines.append(f"| {model_key} | {auc_mean} | {auc_std} | {brier_mean} |")
        else:
            # Quick validation format
            lines.append("| Model | Validation AUC | Validation Brier | Validation ECE |")
            lines.append("|--------|----------------|------------------|----------------|")
            for model_key, val_metrics in cv_results.items():
                auc = val_metrics.get('auc', 'N/A')
                brier = val_metrics.get('brier', 'N/A')
                ece = val_metrics.get('ece', 'N/A')
                if isinstance(auc, float):
                    lines.append(f"| {model_key} | {auc:.4f} | {brier:.4f} | {ece:.4f} |")
                else:
                    lines.append(f"| {model_key} | {auc} | {brier} | {ece} |")
        
        lines.append("")
        lines.append("> **Note:** No true out-of-time validation is possible with this dataset. All temporal fields are relative to the current application. The validation results above use stratified random split with a holdout validation set.")
        lines.append("")
    
    # Calibration
    lines.append("## 8. Calibration (Holdout Test)")
    lines.append("")
    test_metrics = metrics_report.get(model_name, {})
    lines.append(f"- **ECE (Expected Calibration Error):** {test_metrics.get('ece', 'N/A'):.4f}")
    lines.append(f"- **MCE (Max Calibration Error):** {test_metrics.get('mce', 'N/A'):.4f}")
    lines.append(f"- **Calibration Slope:** {test_metrics.get('calibration_slope', 'N/A'):.4f}")
    lines.append(f"- **Calibration Intercept:** {test_metrics.get('calibration_intercept', 'N/A'):.4f}")
    lines.append("")
    
    # Calibration Table
    bin_centers = test_metrics.get('bin_centers', [])
    bin_accuracies = test_metrics.get('bin_accuracies', [])
    bin_counts = test_metrics.get('bin_counts', [])
    if bin_centers and bin_accuracies and bin_counts:
        lines.append("### Calibration Table")
        lines.append("")
        lines.append("| Predicted PD Bucket | Observed Default Rate | Sample Count |")
        lines.append("|---------------------|----------------------|--------------|")
        for c, a, n in zip(bin_centers, bin_accuracies, bin_counts, strict=False):
            lines.append(f"| {c:.2%} | {a:.2%} | {n} |")
        lines.append("")
    
    # PSI
    lines.append("## 9. Population Stability (PSI)")
    lines.append("")
    lines.append(f"- **Test vs Train PSI:** {psi_score:.4f} ({'Stable' if psi_score < 0.1 else 'Moderate shift' if psi_score < 0.25 else 'Significant drift'})")
    lines.append("- **Thresholds:** <0.10 stable, 0.10-0.25 moderate, >0.25 significant")
    lines.append("")
    
    # Decisioning
    if decision_metrics:
        lines.append("## 10. Decision Layer Metrics")
        lines.append("")
        lines.append(f"- **Low threshold:** {config.get('low_threshold', 0.10)}")
        lines.append(f"- **High threshold:** {config.get('high_threshold', 0.30)}")
        lines.append(f"- **Approval rate:** {decision_metrics.get('approval_rate', 'N/A'):.2%}")
        lines.append(f"- **Bad rate (approved):** {decision_metrics.get('bad_rate_approved', 'N/A'):.2%}")
        lines.append(f"- **Default capture rate:** {decision_metrics.get('default_capture_rate', 'N/A'):.2%}")
        lines.append(f"- **False decline rate:** {decision_metrics.get('false_decline_rate', 'N/A'):.2%}")
        lines.append("")
    
    # Expected Loss
    if expected_loss_info:
        lines.append("## 11. Expected Loss")
        lines.append("")
        lines.append(f"- **Total EL (assumed EAD={config.get('ead_default', 'N/A')}, LGD={config.get('lgd_default', 'N/A')}):** {expected_loss_info.get('total_expected_loss', 'N/A'):.2f}")
        lines.append(f"- **Assumptions:** {expected_loss_info.get('assumptions', {}).get('note', 'See config')}")
        lines.append("")
    
    # Explainability
    if shap_comparison is not None:
        lines.append("## 12. Explainability (SHAP vs LogReg)")
        lines.append("")
        lines.append("| Feature | SHAP Rank | LogReg Rank | Rank Diff |")
        lines.append("|---------|-----------|-------------|-----------|")
        for _, row in shap_comparison.head(10).iterrows():
            lines.append(f"| {row['feature']} | {int(row.get('shap_rank', 0))} | {int(row.get('logreg_rank', 0))} | {int(row.get('rank_diff', 0))} |")
        lines.append("")
    
    # Limitations
    lines.append("## 13. Limitations")
    lines.append("")
    default_limits = [
        "Model trained on public/sample data — not representative of production lending portfolio",
        "EAD/LGD assumed constant; actual expected loss requires account-level exposure and recovery data",
        "Fairness analysis limited by available demographic attributes in dataset",
        "**CRITICAL: No true temporal ordering exists in Home Credit data. All date fields are relative to current application. True out-of-time validation is impossible. Validation uses stratified random split + cross-validation only.**",
        "Calibration validated on holdout test only; production drift requires ongoing monitoring",
        "SHAP explanations are post-hoc; causal interpretation not guaranteed",
        "Option A uses only application_train table. Bureau, previous_application, POS_CASH, installments, credit_card_balance not included (Option B would add ~200 features)"
    ]
    for lim in (limitations or default_limits):
        lines.append(f"- {lim}")
    lines.append("")
    
    # Intended Use
    lines.append("## 14. Intended Use")
    lines.append("")
    lines.append("- Portfolio risk segmentation and pricing")
    lines.append("- Automated approval/decline with manual review band")
    lines.append("- Model monitoring and drift detection (PSI)")
    lines.append("- Regulatory model documentation (SR 11-7 style)")
    lines.append("")
    
    # Out of Scope
    lines.append("## 15. Out of Scope")
    lines.append("")
    lines.append("- Individual credit decisions without human oversight for high-risk bands")
    lines.append("- Regulatory capital calculations (requires validated production model)")
    lines.append("- Fairness certification for protected classes")
    lines.append("")
    
    # Monitoring Plan
    lines.append("## 16. Monitoring Plan")
    lines.append("")
    lines.append("- **Monthly:** PSI on model scores and top-10 features")
    lines.append("- **Quarterly:** Full metric re-evaluation on recent data")
    lines.append(f"- **Retrain trigger:** PSI > {config.get('psi_moderate_threshold', 0.25)} or AUC drop > 0.05")
    lines.append("- **Governance:** Model card reviewed at each retrain")
    lines.append("")
    
    # Version
    lines.append("## 17. Version History")
    lines.append("")
    lines.append("| Version | Date | Author | Notes |")
    lines.append("|---------|------|--------|-------|")
    lines.append(f"| 1.0 | {datetime.now().strftime('%Y-%m-%d')} | Pipeline | Initial training (Option A: application_train only) |")
    lines.append("")
    
    return "\n".join(lines)


def save_model_card(content: str, path: str) -> None:
    """Save model card to file."""
    with open(path, "w") as f:
        f.write(content)


def generate_metrics_json(
    metrics_report: dict,
    oot_metrics: dict,
    psi_score: float,
    decision_metrics: dict | None = None,
    expected_loss_info: dict | None = None,
    cv_results: dict | None = None
) -> dict:
    """Generate structured metrics report as JSON."""
    return {
        "test_metrics": metrics_report,
        "oot_metrics": oot_metrics,
        "cv_results": cv_results,
        "psi": psi_score,
        "decision_metrics": decision_metrics,
        "expected_loss": expected_loss_info,
        "generated_at": datetime.now().isoformat()
    }


def save_metrics_report(report: dict, path: str) -> None:
    """Save metrics report to JSON."""
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)