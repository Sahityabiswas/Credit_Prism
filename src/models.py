"""Model training and prediction wrappers for LogReg, Random Forest, and XGBoost."""

from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_logreg(
    X: pd.DataFrame,
    y: pd.Series,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
    calibrate: bool = True
) -> Any:
    """Train Logistic Regression with optional calibration."""
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
        class_weight="balanced"
    )
    model.fit(X, y)
    
    if calibrate:
        # Use isotonic regression for calibration
        model = CalibratedClassifierCV(model, method="isotonic", cv=3)
        model.fit(X, y)
    
    return model


def build_rf(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 10,
    random_state: int = 42,
    calibrate: bool = True
) -> Any:
    """Train Random Forest with optional calibration."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X, y)
    
    if calibrate:
        model = CalibratedClassifierCV(model, method="isotonic", cv=3)
        model.fit(X, y)
    
    return model


def build_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    calibrate: bool = True
) -> Any:
    """Train XGBoost with optional calibration."""
    # Compute scale_pos_weight for class imbalance
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0
    )
    model.fit(X, y)
    
    if calibrate:
        model = CalibratedClassifierCV(model, method="isotonic", cv=3)
        model.fit(X, y)
    
    return model


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict
) -> dict[str, Any]:
    """Train all three candidate models."""
    models = {}
    
    print("Training Logistic Regression...")
    models["logreg"] = build_logreg(
        X_train, y_train,
        C=config.get("logreg_c", 1.0),
        max_iter=config.get("logreg_max_iter", 1000),
        random_state=config.get("random_state", 42)
    )
    
    print("Training Random Forest...")
    models["rf"] = build_rf(
        X_train, y_train,
        n_estimators=config.get("rf_n_estimators", 200),
        max_depth=config.get("rf_max_depth", 10),
        random_state=config.get("random_state", 42)
    )
    
    print("Training XGBoost...")
    models["xgb"] = build_xgb(
        X_train, y_train,
        n_estimators=config.get("xgb_n_estimators", 200),
        max_depth=config.get("xgb_max_depth", 6),
        learning_rate=config.get("xgb_learning_rate", 0.1),
        subsample=config.get("xgb_subsample", 0.8),
        colsample_bytree=config.get("xgb_colsample_bytree", 0.8),
        random_state=config.get("random_state", 42)
    )
    
    return models


def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Get predicted probabilities for class 1 (default)."""
    return model.predict_proba(X)[:, 1]


def save_model(model: Any, path: str) -> None:
    """Save model to disk."""
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    """Load model from disk."""
    return joblib.load(path)


def get_feature_importance(model: Any, feature_names: list) -> pd.DataFrame:
    """Extract feature importance from model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    elif hasattr(model, "base_estimator") and hasattr(model.base_estimator_, "feature_importances_"):
        importances = model.base_estimator_.feature_importances_
    elif hasattr(model, "base_estimator") and hasattr(model.base_estimator_, "coef_"):
        importances = np.abs(model.base_estimator_.coef_[0])
    else:
        return pd.DataFrame({"feature": feature_names, "importance": [np.nan] * len(feature_names)})
    
    return pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)