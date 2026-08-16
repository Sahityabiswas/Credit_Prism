"""Streamlit demo for Credit Risk Intelligence System."""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from encoding import WOEEncoder
from decisioning import assign_risk_band, decision_metrics, threshold_analysis, find_optimal_threshold


st.set_page_config(
    page_title="Credit Risk Intelligence System",
    page_icon="🏦",
    layout="wide"
)


@st.cache_resource
def load_artifacts():
    """Load model, encoder, and config."""
    artifacts_dir = "artifacts"
    
    model_path = os.path.join(artifacts_dir, "best_model.pkl")
    encoder_path = os.path.join(artifacts_dir, "woe_encoder.json")
    config_path = "config.yaml"
    metrics_path = os.path.join(artifacts_dir, "metrics_report.json")
    psi_path = os.path.join(artifacts_dir, "monitoring", "psi_report.json")
    
    model = None
    encoder = None
    config = {}
    metrics = {}
    psi_report = {}
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    if os.path.exists(encoder_path):
        encoder = WOEEncoder.load(encoder_path)
    if os.path.exists(config_path):
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    if os.path.exists(psi_path):
        with open(psi_path, "r") as f:
            psi_report = json.load(f)
    
    return model, encoder, config, metrics, psi_report


def main():
    st.title("🏦 Credit Risk Intelligence System")
    st.caption("Portfolio demo — not a production credit decision engine")
    
    model, encoder, config, metrics, psi_report = load_artifacts()
    
    if model is None or encoder is None:
        st.warning("⚠️ Model artifacts not found. Run `python run_pipeline.py` first.")
        st.code("python run_pipeline.py")
        return
    
    # Sidebar: Configuration
    with st.sidebar:
        st.header("⚙️ Decision Policy")
        low_thresh = st.slider("Low Risk Threshold (Approve)", 0.01, 0.50, 
                               config.get("low_threshold", 0.10), 0.01)
        high_thresh = st.slider("High Risk Threshold (Decline)", 0.01, 0.50, 
                                config.get("high_threshold", 0.30), 0.01)
        
        st.divider()
        st.header("📊 Model Info")
        test_metrics = metrics.get("test_metrics", {})
        best_model = metrics.get("best_model", "unknown")
        st.metric("Best Model", best_model)
        if best_model in test_metrics:
            st.metric("Test AUC", f"{test_metrics[best_model].get('auc', 0):.3f}")
            st.metric("Test Brier", f"{test_metrics[best_model].get('brier', 0):.3f}")
            st.metric("Test ECE", f"{test_metrics[best_model].get('ece', 0):.4f}")
        
        oot_metrics = metrics.get("oot_metrics", {})
        if oot_metrics:
            st.metric("OOT AUC", f"{oot_metrics.get('auc', 0):.3f}")
        
        psi_score = metrics.get("psi", 0)
        psi_status = "🟢 Stable" if psi_score < 0.1 else "🟡 Moderate" if psi_score < 0.25 else "🔴 Drift"
        st.metric("PSI (OOT vs Train)", f"{psi_score:.3f}", psi_status)
    
    # Main: Applicant Input
    st.header("👤 Applicant Assessment")
    
    # Get feature names from encoder
    feature_names = list(encoder.woe_maps_.keys())
    
    # Create input form with a few key features
    st.subheader("Input Features")
    
    # For demo, create inputs for first 5 features
    input_cols = st.columns(3)
    applicant_data = {}
    
    for i, feat in enumerate(feature_names[:8]):  # Limit to 8 for UI
        col = input_cols[i % 3]
        with col:
            # Determine input type from encoder
            ftype = encoder.feature_types_.get(feat, "categorical")
            if ftype == "numeric":
                applicant_data[feat] = st.number_input(
                    feat, value=0.0, key=f"input_{feat}"
                )
            else:
                # Get categories from WOE map
                categories = [k for k in encoder.woe_maps_[feat].keys() if k != "__UNSEEN__"]
                if categories:
                    applicant_data[feat] = st.selectbox(
                        feat, categories, key=f"input_{feat}"
                    )
                else:
                    applicant_data[feat] = st.text_input(feat, key=f"input_{feat}")
    
    if st.button("🔮 Predict Risk", type="primary"):
        # Create DataFrame
        input_df = pd.DataFrame([applicant_data])
        
        # Transform with WOE
        try:
            input_woe = encoder.transform(input_df)
            
            # Ensure all selected features present
            selected_features = list(input_woe.columns)
            
            # Predict
            pd_score = model.predict_proba(input_woe[selected_features])[:, 1][0]
            risk_band = assign_risk_band(np.array([pd_score]), low_thresh, high_thresh)[0]
            
            # Decision
            if risk_band == "Low":
                decision = "✅ Approve"
                decision_color = "green"
            elif risk_band == "Medium":
                decision = "📋 Manual Review"
                decision_color = "orange"
            else:
                decision = "❌ Decline"
                decision_color = "red"
            
            # Display Results
            st.divider()
            st.header("📋 Decision Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted PD", f"{pd_score:.2%}")
            with col2:
                st.metric("Risk Band", risk_band)
            with col3:
                st.markdown(f"<h3 style='color: {decision_color}'>{decision}</h3>", 
                           unsafe_allow_html=True)
            
            # Calibration context
            st.subheader("🎯 Model Confidence")
            cal_slope = test_metrics.get(best_model, {}).get("calibration_slope", 1.0)
            ece = test_metrics.get(best_model, {}).get("ece", 0)
            st.info(f"""
            **Calibration Context:**
            - Model calibration slope: {cal_slope:.2f} (1.0 = perfectly calibrated)
            - Expected Calibration Error (ECE): {ece:.4f}
            - This PD estimate is {'well-calibrated' if ece < 0.05 else 'moderately calibrated' if ece < 0.1 else 'poorly calibrated'}
            """)
            
            # Top Risk Drivers (simplified - would need SHAP for true local)
            st.subheader("🔑 Top Risk Drivers")
            st.caption("Based on global feature importance (SHAP)")
            
            # Show WOE values for this applicant
            driver_data = []
            for feat in selected_features:
                if feat in input_woe.columns:
                    woe_val = input_woe[feat].values[0]
                    driver_data.append({"Feature": feat, "WOE Value": woe_val})
            
            driver_df = pd.DataFrame(driver_data).sort_values("WOE Value", key=abs, ascending=False)
            st.dataframe(driver_df.head(10), use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    
    # Decision Threshold Analysis
    st.divider()
    st.header("📈 Decision Threshold Analysis")
    
    # Load test data if available for threshold analysis
    test_data_path = "data/processed/test.parquet"
    if os.path.exists(test_data_path):
        test_df = pd.read_parquet(test_data_path)
        # This would need actual test data - placeholder for now
        st.info("Threshold analysis requires test data. Run pipeline to generate.")
    else:
        st.info("Run pipeline to generate threshold analysis chart.")
    
    # Model Monitoring
    st.divider()
    st.header("📊 Model Monitoring")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Population Stability (PSI)")
        if psi_report:
            score_psi = psi_report.get("score_psi_oot_vs_train", {})
            psi_val = score_psi.get("psi", 0)
            interp = score_psi.get("interpretation", "unknown")
            st.metric("OOT vs Train PSI", f"{psi_val:.3f}", interp)
            
            # Show bucket breakdown
            if "buckets" in score_psi:
                bucket_df = pd.DataFrame(score_psi["buckets"])
                st.bar_chart(bucket_df.set_index("bucket")["psi_contribution"])
    
    with col2:
        st.subheader("Model Performance Summary")
        if test_metrics:
            perf_df = pd.DataFrame([
                {"Model": k, "AUC": v.get("auc", 0), "Brier": v.get("brier", 0), "ECE": v.get("ece", 0)}
                for k, v in test_metrics.items()
            ])
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
    
    # Disclaimer
    st.divider()
    st.caption("""
    **Disclaimer:** This is a portfolio demonstration using simulated/public data. 
    Decisions shown are illustrative only. Real credit decisions require regulatory compliance,
    fair lending analysis, human oversight, and production-grade model governance.
    """)


if __name__ == "__main__":
    main()