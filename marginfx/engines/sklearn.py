"""
engines/sklearn.py
------------------
Sklearn engine for marginfx.

Provides fit_fn and predict_fn for any scikit-learn compatible model,
including XGBoost, LightGBM, and CatBoost via their sklearn wrappers.

Refit behavior:
    - Every model is cloned and refitted from scratch on each resample.
      See _cold_refit for why warm-starting is wrong here.

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

def _cold_refit(model, X_boot: np.ndarray, y_boot: np.ndarray):
    """
    Refit a fresh copy of the model on a bootstrap resample.

    The refit is deliberately cold. An earlier version warm-started from the
    original fit wherever the library allowed it, which is wrong for a
    resampling diagnostic in two ways. It carries the full-sample fit into
    every replicate, so the spread understates how much the fitted function
    actually moves with the data. And for scikit-learn forests it did nothing
    at all: setting warm_start=True and refitting with n_estimators unchanged
    grows no new trees, so every replicate returned bit-identical predictions
    and the reported dispersion was exactly zero. XGBoost was worse than
    useless rather than inert -- passing the original booster appended another
    round of trees to it, so each replicate held a strictly larger model than
    the one being diagnosed.

    Cloning discards the fitted state and keeps the hyperparameters, which is
    what "refit on the resample" means.

    Parameters
    ----------
    model : object
        The originally fitted model. Not mutated.
    X_boot : np.ndarray
        Resampled features.
    y_boot : np.ndarray
        Resampled targets.

    Returns
    -------
    object
        A model of the same configuration, fitted from scratch on the
        resample.
    """
    from ..learner import clone_estimator

    fresh = clone_estimator(model)
    fresh.fit(X_boot, y_boot)
    return fresh
def make_fit_fn(model) -> Callable:
    """
    Build a fit_fn for a fitted sklearn-compatible model.

    The returned fit_fn clones the model and refits it from scratch on each
    resample.

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
        return _cold_refit(current_model, X_boot, y_boot)

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
            Refits a fresh clone of the model on the bootstrap sample.

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
