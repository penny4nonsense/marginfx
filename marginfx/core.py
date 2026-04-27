"""
core.py
-------
Model-agnostic computation of marginal effects and average marginal effects (AMEs).

This module knows nothing about specific model types. It receives a predict_fn
callable from the engine layer and operates purely on numpy arrays.

Hierarchy:
    me_at_point()       -> marginal effect at a single observation, single feature
    marginal_effects()  -> marginal effect at all observations, single feature (vector)
    ame()               -> average marginal effect, single feature (scalar)
    all_ames()          -> average marginal effect for all features (dict)

Step size (h):
    By default, all_ames() uses adaptive h computed as:
        h_j = max(1e-4, 0.05 * std(X[:, j]))

    For integer-valued features, an additional floor of 0.5 is applied:
        h_j = max(h_j, 0.5)  if feature takes only integer values

    This ensures the finite difference step is meaningful relative to each
    feature's natural scale — critical for tree-based models (XGBoost, random
    forests) where predictions are piecewise constant and a fixed small h may
    never cross a split threshold. The integer floor ensures features like
    education (1-16) and age in whole years reliably cross split boundaries.

    For smooth models (logistic regression, neural nets), the true derivative
    is recovered regardless of h as long as h is reasonably small.

    To use a fixed h for all features, pass h=0.01 (or any float) to all_ames().
    me_at_point(), marginal_effects(), and ame() always take a fixed scalar h.
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional, Union


# ---------------------------------------------------------------------------
# Adaptive step size
# ---------------------------------------------------------------------------

def _compute_adaptive_h(X: np.ndarray) -> np.ndarray:
    """
    Compute per-feature adaptive step sizes for finite differences.

    Formula:
        h_j = max(1e-4, 0.05 * std(X[:, j]))

    For integer-valued features (e.g. education_num, age in whole years),
    an additional floor of 0.5 is applied to ensure the nudge reliably
    crosses tree split thresholds, which occur at integer boundaries:
        h_j = max(h_j, 0.5)  if feature takes only integer values

    This scales the step size to each feature's natural variation:
        - Integer features (e.g. education 1-16):    h = max(0.05*std, 0.5)
        - Unscaled features (e.g. age SD≈13):        h ≈ 0.685
        - Standardized features (SD=1):              h ≈ 0.05
        - Large-scale features (capital_gain SD≈7k): h ≈ 360

    For tree-based models, h must be large enough to cross split thresholds.
    For smooth models, the derivative is recovered accurately for any
    reasonable h.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_obs, n_features).

    Returns
    -------
    np.ndarray
        Per-feature step sizes, shape (n_features,).
    """
    stds = np.std(X, axis=0)
    h = np.maximum(1e-4, 0.05 * stds)

    # For integer-valued features, ensure h >= 0.5 so finite differences
    # reliably cross split thresholds at integer boundaries
    for j in range(X.shape[1]):
        unique_vals = np.unique(X[:, j])
        if len(unique_vals) > 1 and np.all(unique_vals == unique_vals.astype(int)):
            h[j] = max(h[j], 0.5)

    return h


# ---------------------------------------------------------------------------
# Core finite difference computation
# ---------------------------------------------------------------------------

def me_at_point(
    x: np.ndarray,
    feature_idx: int,
    predict_fn: Callable,
    h: float = 1e-4,
    is_categorical: bool = False,
) -> float:
    """
    Compute the marginal effect of a single feature at a single observation.

    For continuous features, uses a central finite difference approximation:
        ME = (f(x + h) - f(x - h)) / (2h)

    For categorical (binary) features, uses a first difference:
        ME = f(x | feature=1) - f(x | feature=0)

    Parameters
    ----------
    x : np.ndarray
        A single observation, shape (n_features,).
    feature_idx : int
        Index of the feature to compute the marginal effect for.
    predict_fn : Callable
        A function that takes an array of shape (n_obs, n_features) and returns
        predictions of shape (n_obs,). Provided by the engine layer.
    h : float
        Step size for finite difference approximation. Default 1e-4.
        For tree models, consider using a larger value. See _compute_adaptive_h.
    is_categorical : bool
        If True, computes a first difference (0 -> 1) instead of a derivative.

    Returns
    -------
    float
        The marginal effect at the given point.
    """
    x = np.array(x, dtype=float)

    if is_categorical:
        x_0 = x.copy()
        x_1 = x.copy()
        x_0[feature_idx] = 0.0
        x_1[feature_idx] = 1.0
        pred_0 = predict_fn(x_0.reshape(1, -1))[0]
        pred_1 = predict_fn(x_1.reshape(1, -1))[0]
        return float(pred_1 - pred_0)
    else:
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[feature_idx] += h
        x_minus[feature_idx] -= h
        pred_plus = predict_fn(x_plus.reshape(1, -1))[0]
        pred_minus = predict_fn(x_minus.reshape(1, -1))[0]
        return float((pred_plus - pred_minus) / (2 * h))


# ---------------------------------------------------------------------------
# Vectorized over dataset
# ---------------------------------------------------------------------------

def marginal_effects(
    X: np.ndarray,
    feature_idx: int,
    predict_fn: Callable,
    h: float = 1e-4,
    is_categorical: bool = False,
) -> np.ndarray:
    """
    Compute the marginal effect of a single feature at every observation in X.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (n_obs, n_features).
    feature_idx : int
        Index of the feature to compute marginal effects for.
    predict_fn : Callable
        A function that takes (n_obs, n_features) and returns (n_obs,).
    h : float
        Step size for finite difference approximation.
    is_categorical : bool
        If True, computes first differences instead of derivatives.

    Returns
    -------
    np.ndarray
        Array of marginal effects, shape (n_obs,).
    """
    X = np.array(X, dtype=float)

    if is_categorical:
        X_0 = X.copy()
        X_1 = X.copy()
        X_0[:, feature_idx] = 0.0
        X_1[:, feature_idx] = 1.0
        return predict_fn(X_1) - predict_fn(X_0)
    else:
        X_plus = X.copy()
        X_minus = X.copy()
        X_plus[:, feature_idx] += h
        X_minus[:, feature_idx] -= h
        return (predict_fn(X_plus) - predict_fn(X_minus)) / (2 * h)


# ---------------------------------------------------------------------------
# Average marginal effect — single feature
# ---------------------------------------------------------------------------

def ame(
    X: np.ndarray,
    feature_idx: int,
    predict_fn: Callable,
    h: float = 1e-4,
    is_categorical: bool = False,
) -> float:
    """
    Compute the average marginal effect (AME) of a single feature.

    The AME is the mean of the pointwise marginal effects across all observations.
    This is the scalar quantity analogous to an OLS coefficient.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (n_obs, n_features).
    feature_idx : int
        Index of the feature.
    predict_fn : Callable
        Prediction function from the engine layer.
    h : float
        Step size for finite differences.
    is_categorical : bool
        If True, computes first differences.

    Returns
    -------
    float
        The average marginal effect.
    """
    me_vector = marginal_effects(X, feature_idx, predict_fn, h, is_categorical)
    return float(np.mean(me_vector))


# ---------------------------------------------------------------------------
# Average marginal effects — all features
# ---------------------------------------------------------------------------

def all_ames(
    X: np.ndarray,
    predict_fn: Callable,
    feature_names: Optional[list] = None,
    categorical_features: Optional[list] = None,
    h: Union[float, str] = 'adaptive',
) -> dict:
    """
    Compute average marginal effects for all features.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (n_obs, n_features).
    predict_fn : Callable
        Prediction function from the engine layer.
    feature_names : list, optional
        Names for each feature. Defaults to ['x0', 'x1', ...].
    categorical_features : list, optional
        List of feature indices (or names) that are categorical/binary.
        These will use first differences instead of derivatives.
    h : float or 'adaptive'
        Step size for finite differences. Default 'adaptive' computes
        h_j = max(1e-4, 0.05 * std(X[:, j])) per feature, with an
        additional floor of 0.5 for integer-valued features.
        Pass a float to use a fixed step size for all features.

    Returns
    -------
    dict
        Dictionary mapping feature name -> AME estimate.
    """
    X = np.array(X, dtype=float)
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]

    if categorical_features is None:
        categorical_features = []

    # Normalize categorical_features to indices
    cat_indices = set()
    for cf in categorical_features:
        if isinstance(cf, str):
            if cf in feature_names:
                cat_indices.add(feature_names.index(cf))
        elif isinstance(cf, int):
            cat_indices.add(cf)

    # Compute adaptive h per feature or use fixed scalar
    if h == 'adaptive':
        h_values = _compute_adaptive_h(X)
    else:
        h_values = np.full(n_features, float(h))

    results = {}
    for idx, name in enumerate(feature_names):
        is_cat = idx in cat_indices
        results[name] = ame(X, idx, predict_fn, h_values[idx], is_cat)

    return results


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class MarginfxResult:
    """
    Container for marginfx estimation results.

    Holds AME estimates and optionally standard errors and confidence intervals
    from the bootstrap. Provides tidy DataFrame output and a summary table
    analogous to the broom package in R.

    Attributes
    ----------
    estimates : dict
        Feature name -> AME estimate.
    std_errors : dict or None
        Feature name -> bootstrap standard error.
    conf_int : dict or None
        Feature name -> (lower, upper) confidence interval tuple.
    n_obs : int
        Number of observations used.
    n_bootstrap : int or None
        Number of bootstrap replicates used.
    alpha : float
        Significance level for confidence intervals. Default 0.05.
    """

    def __init__(
        self,
        estimates: dict,
        std_errors: Optional[dict] = None,
        conf_int: Optional[dict] = None,
        n_obs: int = 0,
        n_bootstrap: Optional[int] = None,
        alpha: float = 0.05,
    ):
        self.estimates = estimates
        self.std_errors = std_errors
        self.conf_int = conf_int
        self.n_obs = n_obs
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha

    def tidy(self) -> pd.DataFrame:
        """
        Return results as a tidy DataFrame, analogous to broom::tidy() in R.

        Returns
        -------
        pd.DataFrame
            Columns: term, estimate, std_error, statistic, p_value,
                     conf_low, conf_high (last four only if bootstrap was run).
        """
        rows = []
        for feature, est in self.estimates.items():
            row = {"term": feature, "estimate": est}

            if self.std_errors and feature in self.std_errors:
                se = self.std_errors[feature]
                row["std_error"] = se
                row["statistic"] = est / se if se > 0 else np.nan
                # Normal approximation p-value (two-tailed)
                from scipy import stats
                row["p_value"] = 2 * (1 - stats.norm.cdf(abs(row["statistic"])))

            if self.conf_int and feature in self.conf_int:
                row["conf_low"] = self.conf_int[feature][0]
                row["conf_high"] = self.conf_int[feature][1]

            rows.append(row)

        return pd.DataFrame(rows)

    def summary(self) -> None:
        """
        Print a formatted summary table to stdout.
        """
        df = self.tidy()
        print("=" * 65)
        print("marginfx: Average Marginal Effects")
        print("=" * 65)
        print(f"Observations: {self.n_obs}")
        if self.n_bootstrap:
            print(f"Bootstrap replicates: {self.n_bootstrap}")
            print(f"Confidence level: {int((1 - self.alpha) * 100)}%")
        print("-" * 65)
        print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        print("=" * 65)

    def __repr__(self) -> str:
        n = len(self.estimates)
        has_se = self.std_errors is not None
        return (
            f"MarginfxResult("
            f"features={n}, "
            f"bootstrap={'yes' if has_se else 'no'}, "
            f"n_obs={self.n_obs})"
        )
