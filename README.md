# Credit Risk Intelligence System

End-to-end, explainable credit-risk modeling pipeline with custom WOE/IV encoding, calibration-first evaluation, PSI monitoring, and auto-generated model cards.

---

## 🎯 Problem → Approach → Key Results → Business Decision

| Problem | Approach | Key Results | Business Decision |
|---------|----------|-------------|-------------------|
| Credit risk models optimize AUC only, ignoring calibration, stability, and decision utility | Custom WOE/IV pipeline, LogReg/RF/XGBoost with isotonic calibration, stratified split + holdout, PSI drift monitoring, auto model card | **Best: Random Forest** — AUC 0.735, Brier 0.069, ECE 0.0015, PSI 0.0003 (stable) | Risk bands PD 10%/30% → 74% approval, 4.7% bad rate, 8% default capture |

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python run_pipeline.py          # Auto-downloads Home Credit data from Hugging Face
cat artifacts/model_card.md     # View results
streamlit run app/streamlit_app.py  # Interactive demo
```

---

## 🏗️ Architecture

```
credit-risk-intelligence/
├── config.yaml                 # All parameters
├── run_pipeline.py             # Single entry point
├── src/
│   ├── data_ingestion.py       # HF loading + cleaning
│   ├── encoding.py             # WOE encoder (from scratch)
│   ├── feature_selection.py    # IV, correlation, VIF
│   ├── models.py               # LogReg, RF, XGBoost + calibration
│   ├── evaluate.py             # AUC, PR-AUC, Gini, KS, Brier, ECE
│   ├── decisioning.py          # Risk bands, thresholds
│   ├── expected_loss.py        # PD × EAD × LGD
│   ├── explain.py              # SHAP global + local
│   ├── monitoring.py           # PSI for scores/features
│   └── model_card.py           # Auto-generated model card
├── tests/                      # Deterministic math tests
├── artifacts/                  # Generated outputs (gitignored)
├── app/streamlit_app.py        # Demo UI
└── .github/workflows/ci.yml    # Lint + tests
```

---

## 🔑 Key Differentiators

1. **Custom WOE + IV** — Built from log-odds math, not `category_encoders`
2. **Calibration first-class** — Brier, reliability curves, ECE/MCE, calibration slope/intercept
3. **PSI monitoring** — Score/feature drift detection; PSI > 0.25 triggers review
4. **SHAP cross-validated against LogReg** — Compare black-box drivers with interpretable WOE coefficients
5. **Auto-generated model card** — Pipeline outputs → structured card (use cases, limitations, monitoring, retrain triggers)

---

## 🏦 Home Credit Dataset

| Aspect | Value |
|--------|-------|
| Rows | 307,511 |
| Target | `TARGET` (8% default rate) |
| Features | 122 (after cleaning) |
| Split | Stratified random (no true temporal ordering) |
| Test AUC | 0.735 |

**Cleaning applied:** Drop `SK_ID_CURR`; `DAYS_EMPLOYED=365243`→NaN; drop `CODE_GENDER="XNA"`; `DAYS_LAST_PHONE_CHANGE=0`→NaN; `REGION_RATING_CLIENT=-1`→NaN; target→int.

**Results:** Best model Random Forest — AUC 0.735, Brier 0.069, ECE 0.0015, PSI 0.0003. 24 features selected from 121 (IV > 0.02, corr < 0.8, VIF < 10). Top: `EXT_SOURCE_3`, `EXT_SOURCE_2`, `DAYS_EMPLOYED`, `AMT_GOODS_PRICE`.

**Configuration:** See `config.yaml` (all parameters: data, feature engineering, modeling, evaluation, decision thresholds, monitoring, output paths).

---

## 🧪 Testing

```bash
pytest tests/ -v    # Deterministic math only (WOE, IV, PSI, leakage)
```

---

## 📋 Requirements

- Python 3.10+
- `pip install -r requirements.txt` (pandas, numpy, scikit-learn, xgboost, shap, scipy, pyyaml, streamlit, matplotlib, seaborn, joblib, datasets)

---

## 📄 License

MIT — free for learning and portfolio use.