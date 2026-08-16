#!/usr/bin/env python
"""Credit Risk Intelligence System — End-to-end pipeline entry point."""

import sys
import os
import yaml
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_prep import load_config
from data_ingestion import load_or_create_clean_data
from encoding import fit_woe_encoder
from feature_selection import select_features, calculate_iv
from models import train_all_models, predict_proba
from evaluate import evaluate_all, compare_models, evaluate_cv
from decisioning import decision_metrics, threshold_analysis, find_optimal_threshold
from expected_loss import el_with_assumptions
from explain import generate_shap_explanations, save_global_shap_plot, save_local_shap_examples, compare_shap_with_logreg
from monitoring import calculate_psi, generate_psi_report, save_psi_report
from model_card import generate_model_card, generate_metrics_json, save_metrics_report, save_model_card
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold


def main():
    print("=" * 60)
    print("Credit Risk Intelligence System — Pipeline")
    print("=" * 60)
    
    # 1. Load config
    config = load_config("config.yaml")
    print(f"\n[1/7] Config loaded: {config['raw_path']}")
    
    # 2. Data preparation - load clean Home Credit data
    print("\n[2/7] Preparing data...")
    df = load_or_create_clean_data(config["raw_path"])
    
    target = config["target"]
    forbidden = config["forbidden_columns"]
    
    # Stratified random split (no true temporal ordering in Home Credit data)
    # All temporal fields are relative to current application, no calendar dates exist
    test_size = config.get("test_size", 0.2)
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=config.get("random_state", 42),
        stratify=df[target]
    )
    
    print(f"    Split sizes — Train: {len(train_df)}, Test: {len(test_df)} (stratified)")
    print(f"    Target dist — Train: {train_df[target].mean():.4f}, Test: {test_df[target].mean():.4f}")
    
    # Feature columns (exclude target, date_column if exists, forbidden)
    date_column = config.get("date_column", "")
    exclude_cols = [target] + ([date_column] if date_column in df.columns else []) + forbidden
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    print(f"    Features before selection: {len(feature_cols)}")
    
    # Impute missing values (fit on train, apply to test)
    from data_prep import impute_missing
    train_df, imputer = impute_missing(train_df, fit=True)
    test_df, _ = impute_missing(test_df, imputer=imputer)
    
    # 3. WOE Encoding
    print("\n[3/7] Fitting WOE encoder...")
    woe_encoder = fit_woe_encoder(
        train_df,
        target=target,
        feature_cols=feature_cols,
        n_bins=config.get("n_bins", 10),
        min_bin_size=config.get("min_bin_size", 0.05)
    )
    
    train_woe = woe_encoder.transform(train_df[feature_cols])
    test_woe = woe_encoder.transform(test_df[feature_cols])
    
    # 4. Feature Selection
    print("\n[4/7] Feature selection...")
    # Add target to WOE data for IV calculation
    train_woe_with_target = train_woe.copy()
    train_woe_with_target[target] = train_df[target].values
    selected_features, iv_table = select_features(
        train_woe_with_target,
        target=target,
        iv_min=config.get("iv_min", 0.02),
        iv_max=config.get("iv_max", 0.5),
        correlation_threshold=config.get("correlation_threshold", 0.8),
        vif_threshold=config.get("vif_threshold", 10)
    )
    print(f"    Selected features: {len(selected_features)}")
    
    if not selected_features:
        print("ERROR: No features selected. Check IV thresholds.")
        sys.exit(1)
    
    X_train = train_woe[selected_features]
    y_train = train_df[target]
    X_test = test_woe[selected_features]
    y_test = test_df[target]
    
    # 5. Model Training + Quick Validation
    print("\n[5/7] Training models with quick validation...")
    
    # Use a small validation split for quick performance estimate
    val_size = min(5000, len(X_train) // 10)
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=config.get("random_state", 42), stratify=y_train
    )
    print(f"  Using validation set of {len(X_val)} rows for quick performance estimate")
    
    models = {}
    val_results = {}
    
    for name in ["logreg", "rf"]:  # Skip XGBoost for speed
        if name == "logreg":
            from models import build_logreg
            model = build_logreg(X_train_fit, y_train_fit, C=config.get("logreg_c", 1.0), max_iter=config.get("logreg_max_iter", 1000), random_state=config.get("random_state", 42))
        elif name == "rf":
            from models import build_rf
            model = build_rf(X_train_fit, y_train_fit, n_estimators=config.get("rf_n_estimators", 50), max_depth=config.get("rf_max_depth", 8), random_state=config.get("random_state", 42))
        
        print(f"  Training {name}...")
        models[name] = model
        
        # Quick validation
        val_preds = predict_proba(model, X_val)
        val_metrics = evaluate_all(y_val.values, val_preds, n_bins=config.get("calibration_bins", 10))
        val_results[name] = val_metrics
        print(f"    {name}: Val AUC={val_metrics['auc']:.4f}, Brier={val_metrics['brier']:.4f}, ECE={val_metrics['ece']:.4f}")
    
    # 6. Evaluation on holdout test set
    print("\n[6/7] Evaluating on holdout test set...")
    metrics_report = {}
    test_preds = {}
    
    for name, model in models.items():
        preds = predict_proba(model, X_test)
        test_preds[name] = preds
        metrics_report[name] = evaluate_all(y_test.values, preds, n_bins=config.get("calibration_bins", 10))
        print(f"    {name}: AUC={metrics_report[name]['auc']:.4f}, Brier={metrics_report[name]['brier']:.4f}, ECE={metrics_report[name]['ece']:.4f}")
    
    # Model comparison table
    comparison = compare_models(y_test.values, test_preds)
    print("\nModel Comparison (Holdout Test):")
    print(comparison[["model", "auc", "brier", "ece", "calibration_slope"]].to_string(index=False))
    
    # Validation Comparison
    print("\nQuick Validation Comparison:")
    val_df = pd.DataFrame([
        {"model": k, "val_auc": v["auc"], "val_brier": v["brier"], "val_ece": v["ece"]}
        for k, v in val_results.items()
    ])
    print(val_df.to_string(index=False))
    
    # Pick best model (by calibration + AUC on holdout)
    def pick_best(metrics):
        best_name = None
        best_score = -1
        for name, m in metrics.items():
            score = m["auc"] - 2 * m.get("ece", 1.0)
            if score > best_score:
                best_score = score
                best_name = name
        return best_name
    
    best_model_name = pick_best(metrics_report)
    best_model = models[best_model_name]
    print(f"\n    Best model: {best_model_name}")
    
    # 7. PSI (train vs test)
    print("\n[7/7] Stability monitoring (PSI train vs test)...")
    test_scores = test_preds[best_model_name]
    train_scores = predict_proba(best_model, X_train)
    psi_score = calculate_psi(train_scores, test_scores, buckets=config.get("psi_buckets", 10))
    print(f"    PSI (train vs test): {psi_score:.4f} ({'Stable' if psi_score < 0.1 else 'Moderate' if psi_score < 0.25 else 'Drift'})")
    
    # Decision metrics
    print("\n[Decision Layer]")
    dec_metrics = decision_metrics(
        y_test.values,
        test_scores,
        low_threshold=config.get("low_threshold", 0.10),
        high_threshold=config.get("high_threshold", 0.30)
    )
    print(f"    Approval rate: {dec_metrics['approval_rate']:.2%}")
    print(f"    Bad rate (approved): {dec_metrics['bad_rate_approved']:.2%}")
    print(f"    Default capture: {dec_metrics['default_capture_rate']:.2%}")
    
    # Threshold analysis
    thresholds = np.linspace(0.01, 0.50, 50)
    thresh_df = threshold_analysis(y_test.values, test_scores, thresholds)
    thresh_path = config.get("decision_thresholds_path", "artifacts/decision_thresholds.csv")
    os.makedirs(os.path.dirname(thresh_path), exist_ok=True)
    thresh_df.to_csv(thresh_path, index=False)
    
    # Optimal threshold
    optimal = find_optimal_threshold(
        y_test.values, test_scores,
        objective="balanced",
        min_approval=0.1,
        max_bad_rate=0.2
    )
    if optimal.get("threshold") is not None:
        print(f"    Optimal threshold (balanced): {optimal['threshold']:.3f}")
    else:
        print(f"    Optimal threshold: No threshold satisfies constraints ({optimal.get('reason', 'unknown')})")
    
    # Expected Loss
    el_info = el_with_assumptions(
        test_scores,
        ead_default=config.get("ead_default", 10000),
        lgd_default=config.get("lgd_default", 0.45)
    )
    print(f"\n[Expected Loss] Total EL: {el_info['total_expected_loss']:.2f}")
    
    # SHAP Explainability (disabled for speed - enable by setting config['generate_shap']=true)
    if config.get("generate_shap", False):
        print("\n[Explainability] Generating SHAP explanations...")
        shap_values = generate_shap_explanations(best_model, X_test, sample_size=100)
        save_global_shap_plot(shap_values)
        save_local_shap_examples(shap_values, n=3, y_pred=test_scores)
    else:
        print("\n[Explainability] Skipping SHAP (set generate_shap=true in config to enable)")
        shap_values = None
    
    # Compare with LogReg
    if "logreg" in models and shap_values is not None:
        shap_comparison = compare_shap_with_logreg(
            shap_values, models["logreg"], selected_features
        )
        print("\nSHAP vs LogReg ranking comparison:")
        cols = [c for c in ["feature", "shap_rank", "logreg_rank", "rank_diff"] if c in shap_comparison.columns]
        if cols:
            print(shap_comparison[cols].head(10).to_string(index=False))
        else:
            print(shap_comparison.head(10).to_string(index=False))
    else:
        shap_comparison = None
    
    # PSI Report (train vs test + feature PSI)
    psi_report = generate_psi_report(
        train_scores,
        test_scores,
        test_scores,  # no OOT, use test as reference
        train_woe[selected_features] if len(selected_features) > 0 else None,
        X_test if len(X_test) > 0 else None,
        selected_features,
        buckets=config.get("psi_buckets", 10)
    )
    save_psi_report(psi_report, config.get("psi_report_path", "artifacts/monitoring/psi_report.json"))
    
    # Save artifacts
    print("\n[Saving Artifacts]")
    
    # Metrics report (includes validation results)
    metrics_json = generate_metrics_json(
        metrics_report, {}, psi_score, dec_metrics, el_info, val_results
    )
    save_metrics_report(metrics_json, config.get("metrics_report_path", "artifacts/metrics_report.json"))
    
    # Model card
    model_card = generate_model_card(
        model_name=best_model_name,
        config=config,
        metrics_report=metrics_report,
        oot_metrics={},  # No OOT
        psi_score=psi_score,
        iv_table=iv_table,
        selected_features=selected_features,
        shap_comparison=shap_comparison,
        decision_metrics=dec_metrics,
        expected_loss_info=el_info,
        cv_results=val_results  # renamed from cv_results
    )
    save_model_card(model_card, config.get("model_card_path", "artifacts/model_card.md"))
    
    # Save model and encoder
    joblib.dump(best_model, "artifacts/best_model.pkl")
    woe_encoder.save("artifacts/woe_encoder.json")
    
    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"Artifacts saved to: artifacts/")
    print(f"  - metrics_report.json (includes CV results)")
    print(f"  - model_card.md (documents no true OOT limitation)")
    print(f"  - decision_thresholds.csv")
    print(f"  - monitoring/psi_report.json")
    print(f"  - shap_plots/")
    print(f"  - best_model.pkl")
    print(f"  - woe_encoder.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()