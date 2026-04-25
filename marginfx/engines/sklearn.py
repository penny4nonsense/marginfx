"""
engines/sklearn.py
------------------
Sklearn engine for marginfx.

Provides fit_fn and predict_fn for any scikit-learn compatible model,
including XGBoost, LightGBM, and CatBoost via their sklearn wrappers.

Warm-start behavior:
    - Models with warm_start attribute: warm-started automatically
    - XGBoost sklearn wrapper: warm-started via xgb_model parameter
    - All other models: refitted cold, silently

Prediction behavior:
    - Models with predict_proba: uses predict_proba[:, 1] (binary classification)
    - All other models: uses predict (regression)
"""

import numpy as np
import copy
from typing import Callable, Tuple


# ---------------------------------------------------------------------------
# Predict function
# ---------------------------------------------------------------------------

def make_predict_fn(model) -> Callable:
    """
    Build a predict_fn for a fitted sklearn-compatible model.

    Autodetects classification vs regression:
        - If model has predict_proba: returns P(y=1) for binary classification
        - Otherwise: returns predict() output directly

    Parameters
    ----------
    model : fitted sklearn-compatible model
        Any model with a .predict() method. Optionally .predict_proba().

    Returns
    -------
    Callable
        predict_fn(X) -> np.ndarray of shape (n_obs,)
    """
    if hasattr(model, 'predict_proba'):
        def predict_fn(X):
            proba = model.predict_proba(X)
            # Binary classification: return P(y=1)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            # Multiclass: return full probability matrix
            # AMEs computed per class — future feature, return as-is for now
            return proba
    else:
        def predict_fn(X):
            return model.predict(X)

    return predict_fn


# ---------------------------------------------------------------------------
# Fit function
# ---------------------------------------------------------------------------

def _is_xgboost(model) -> bool:
    """Check if model is an XGBoost sklearn wrapper."""
    try:
        import xgboost as xgb
        return isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor))
    except ImportError:
        return False


def _is_lightgbm(model) -> bool:
    """Check if model is a LightGBM sklearn wrapper."""
    try:
        import lightgbm as lgb
        return isinstance(model, (lgb.LGBMClassifier, lgb.LGBMRegressor))
    except ImportError:
        return False


def _warm_start_refit(model, X_boot: np.ndarray, y_boot: np.ndarray):
    """
    Refit a model with warm-start if supported, otherwise refit cold.

    Warm-start support:
        - sklearn models with warm_start attribute
        - XGBoost via xgb_model parameter
        - LightGBM via init_model parameter

    Parameters
    ----------
    model : fitted model
        Original fitted model used as warm-start initialization.
    X_boot : np.ndarray
        Bootstrap sample features.
    y_boot : np.ndarray
        Bootstrap sample targets.

    Returns
    -------
    fitted model
        New model fitted on bootstrap sample.
    """

    # --- XGBoost warm-start ---
    if _is_xgboost(model):
        new_model = copy.copy(model)
        # XGBoost warm-starts via xgb_model parameter in fit()
        new_model.fit(
            X_boot,
            y_boot,
            xgb_model=model.get_booster(),
        )
        return new_model

    # --- LightGBM warm-start ---
    if _is_lightgbm(model):
        new_model = copy.copy(model)
        new_model.fit(
            X_boot,
            y_boot,
            init_model=model.booster_,
        )
        return new_model

    # --- Sklearn warm_start attribute ---
    if hasattr(model, 'warm_start'):
        new_model = copy.deepcopy(model)
        new_model.warm_start = True
        new_model.fit(X_boot, y_boot)
        return new_model

    # --- Cold refit fallback ---
    new_model = copy.deepcopy(model)
    new_model.fit(X_boot, y_boot)
    return new_model


def make_fit_fn(model) -> Callable:
    """
    Build a fit_fn for a fitted sklearn-compatible model.

    The returned fit_fn warm-starts from the original model where possible,
    falling back to cold refit silently for unsupported models.

    Parameters
    ----------
    model : fitted sklearn-compatible model
        Original fitted model.

    Returns
    -------
    Callable
        fit_fn(model, X_boot, y_boot) -> fitted_model
    """
    def fit_fn(current_model, X_boot, y_boot):
        return _warm_start_refit(current_model, X_boot, y_boot)

    return fit_fn


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_engine(model) -> Tuple[Callable, Callable]:
    """
    Get predict_fn and fit_fn for a sklearn-compatible model.

    This is the main entry point called by the marginfx API layer.

    Parameters
    ----------
    model : fitted sklearn-compatible model
        Any model with a .predict() method.

    Returns
    -------
    Tuple[Callable, Callable]
        (predict_fn, fit_fn)

        predict_fn(X) -> np.ndarray
            Returns predictions for input X.

        fit_fn(model, X_boot, y_boot) -> fitted_model
            Refits model on bootstrap sample, warm-starting where possible.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from engines.sklearn import get_engine
    >>> from bootstrap import bootstrap_ames
    >>>
    >>> model = RandomForestClassifier().fit(X_train, y_train)
    >>> predict_fn, fit_fn = get_engine(model)
    >>> result = bootstrap_ames(model, X, y, fit_fn, predict_fn)
    >>> result.summary()
    """
    predict_fn = make_predict_fn(model)
    fit_fn = make_fit_fn(model)
    return predict_fn, fit_fn
