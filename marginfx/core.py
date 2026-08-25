"""
core.py
-------
Model-agnostic computation of window average marginal effects.

This module knows nothing about specific model types. It receives a predict_fn
callable from the engine layer and operates purely on numpy arrays.

Estimand
--------
The target is the *window average marginal effect* at scale h (Definition 1 of
the paper):

    theta_{j,h} = E[ w(X) * (f(X + h e_j) - f(X - h e_j)) / (2h) ]

where w is a trimming weight supported on

    Omega_{j,h} = { x : x + h e_j in Omega and x - h e_j in Omega }

Two consequences of this definition drive the code below.

1.  h is part of the estimand, not a numerical tolerance. The centered
    difference is the *defining operation* of the target, computed exactly --
    not an O(h^2) approximation to a derivative. h must therefore be chosen,
    held fixed, and reported. It is recorded on MarginfxResult and printed by
    summary().

2.  Trimming is mandatory by default. Points within h_j of the boundary of the
    observed support would require evaluating f at x_ij +/- h_j outside the
    support, where the regression function is not identified. Those points get
    weight zero. The trimmed shell carries probability mass O(h_j).

theta_{j,h} converges to the classical average marginal effect E[df/dx_j] as
h -> 0 whenever f has one weak derivative.

Step size (h)
-------------
Default is 'adaptive':

    h_j = max(1e-4, 0.05 * std(X[:, j]))

The floor of 1e-4 guards against degenerate features with std ~ 0. With the
default multiplier, h_j asks how the prediction responds to a displacement of
a twentieth of a standard deviation.

Note there is deliberately NO integer floor. An earlier version of this module
applied h_j = max(h_j, 0.5) to integer-valued features so that finite
differences would cross tree split thresholds. Under the window estimand that
is not a numerical convenience but a silent redefinition of the target, so it
has been removed.
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional, Union


# ---------------------------------------------------------------------------
# Step size
# ---------------------------------------------------------------------------

def compute_adaptive_h(X: np.ndarray) -> np.ndarray:
    """
    Compute per-feature adaptive step sizes.

        h_j = max(1e-4, 0.05 * std(X[:, j]))

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
    return np.maximum(1e-4, 0.05 * stds)


def resolve_h(X: np.ndarray, h: Union[float, str, np.ndarray]) -> np.ndarray:
    """
    Normalize the `h` argument to a per-feature array.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_obs, n_features).
    h : float, 'adaptive', or np.ndarray
        Step size specification. 'adaptive' uses compute_adaptive_h.

    Returns
    -------
    np.ndarray
        Per-feature step sizes, shape (n_features,).
    """
    n_features = X.shape[1]
    if isinstance(h, str):
        if h != 'adaptive':
            raise ValueError(f"h must be a float, an array, or 'adaptive'; got {h!r}")
        return compute_adaptive_h(X)
    h_arr = np.asarray(h, dtype=float)
    if h_arr.ndim == 0:
        return np.full(n_features, float(h_arr))
    if h_arr.shape != (n_features,):
        raise ValueError(
            f"h array has shape {h_arr.shape}, expected ({n_features},)"
        )
    return h_arr


# ---------------------------------------------------------------------------
# Support bounds and trimming
# ---------------------------------------------------------------------------

def support_bounds(X: np.ndarray) -> tuple:
    """
    Per-feature bounds of the observed support.

    These define the empirical analog of Omega. They must be computed once on
    the full sample and then held fixed -- the trimming weight is part of the
    estimand, so it cannot be allowed to vary across folds or bootstrap
    replicates.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_obs, n_features).

    Returns
    -------
    tuple of np.ndarray
        (lower, upper), each of shape (n_features,).
    """
    return np.min(X, axis=0), np.max(X, axis=0)


def trimming_weight(
    X: np.ndarray,
    feature_idx: int,
    h: float,
    bounds: Optional[tuple] = None,
) -> np.ndarray:
    """
    Trimming weight w for a single feature.

    Returns 1.0 for observations whose both shifted evaluation points stay
    inside the observed support of feature j, and 0.0 otherwise:

        w_i = 1{ x_ij + h <= upper_j  and  x_ij - h >= lower_j }

    This is the indicator of Omega_{j,h}. Without it, f is evaluated outside
    the support, where the regression function is unidentified and the fitted
    model is silently extrapolating.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_obs, n_features).
    feature_idx : int
        Index of the feature being perturbed.
    h : float
        Step size for this feature.
    bounds : tuple of np.ndarray, optional
        (lower, upper) from support_bounds(). If None, computed from X.
        Pass explicitly when X is a subsample, so that the estimand does not
        drift with the subsample.

    Returns
    -------
    np.ndarray
        Weights in {0.0, 1.0}, shape (n_obs,).
    """
    if bounds is None:
        bounds = support_bounds(X)
    lower, upper = bounds
    xj = X[:, feature_idx]
    inside = (xj + h <= upper[feature_idx]) & (xj - h >= lower[feature_idx])
    return inside.astype(float)


# ---------------------------------------------------------------------------
# The difference operator
# ---------------------------------------------------------------------------

def window_difference(
    X: np.ndarray,
    feature_idx: int,
    predict_fn: Callable,
    h: float,
) -> np.ndarray:
    """
    Centered difference operator D_h applied to the prediction function.

        D_h f(x_i) = (f(x_i + h e_j) - f(x_i - h e_j)) / (2h)

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (n_obs, n_features).
    feature_idx : int
        Index of the feature to perturb.
    predict_fn : Callable
        Takes (n_obs, n_features), returns (n_obs,). For binary
        classification this must return P(Y=1 | x), not a class label --
        class labels are discrete and their differences are identically zero.
    h : float
        Step size.

    Returns
    -------
    np.ndarray
        Difference at each observation, shape (n_obs,).
    """
    X_plus = X.copy()
    X_minus = X.copy()
    X_plus[:, feature_idx] += h
    X_minus[:, feature_idx] -= h
    return (predict_fn(X_plus) - predict_fn(X_minus)) / (2.0 * h)


def contrast(
    X: np.ndarray,
    feature_idx: int,
    predict_fn: Callable,
) -> np.ndarray:
    """
    First difference for a binary or categorical feature.

        m_ij = f(x_i | x_j = 1) - f(x_i | x_j = 0)

    No step size and no trimming apply: the perturbation is between two levels
    that both lie in the support by construction.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (n_obs, n_features).
    feature_idx : int
        Index of the binary feature.
    predict_fn : Callable
        Takes (n_obs, n_features), returns (n_obs,).

    Returns
    -------
    np.ndarray
        Contrast at each observation, shape (n_obs,).
    """
    X_0 = X.copy()
    X_1 = X.copy()
    X_0[:, feature_idx] = 0.0
    X_1[:, feature_idx] = 1.0
    return predict_fn(X_1) - predict_fn(X_0)


def pointwise_effects(
    X: np.ndarray,
    feature_idx: int,
    predict_fn: Callable,
    h: float,
    is_categorical: bool = False,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Weighted pointwise effect for one feature -- the summand of the plug-in.

    Continuous: w_i * D_h f(x_i).  Categorical: the contrast (w_i == 1).

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (n_obs, n_features).
    feature_idx : int
        Feature index.
    predict_fn : Callable
        Prediction function.
    h : float
        Step size. Ignored when is_categorical is True.
    is_categorical : bool
        If True, use the contrast instead of the difference operator.
    weights : np.ndarray, optional
        Trimming weights, shape (n_obs,). Defaults to all ones.

    Returns
    -------
    np.ndarray
        Weighted effects, shape (n_obs,).
    """
    if is_categorical:
        return contrast(X, feature_idx, predict_fn)

    effects = window_difference(X, feature_idx, predict_fn, h)
    if weights is None:
        return effects
    return weights * effects


# ---------------------------------------------------------------------------
# Plug-in estimator
# ---------------------------------------------------------------------------

def plugin_ame(
    X: np.ndarray,
    feature_idx: int,
    predict_fn: Callable,
    h: float,
    is_categorical: bool = False,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Plug-in estimate of the window AME for a single feature.

        theta_hat = (1/n) sum_i w_i * D_h f(x_i)

    Note the average is over all n observations, NOT renormalized by sum(w).
    theta_{j,h} is defined as E[w(X) D_h f(X)], so renormalizing would target a
    different functional (the effect conditional on being untrimmed).

    This estimator is biased by the learner's regularization; see dml.py for
    the debiased version that the paper uses for inference.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (n_obs, n_features).
    feature_idx : int
        Feature index.
    predict_fn : Callable
        Prediction function.
    h : float
        Step size.
    is_categorical : bool
        If True, use the contrast.
    weights : np.ndarray, optional
        Trimming weights.

    Returns
    -------
    float
        The plug-in window average marginal effect.
    """
    return float(np.mean(
        pointwise_effects(X, feature_idx, predict_fn, h, is_categorical, weights)
    ))


def resolve_categorical(
    categorical_features: Optional[list],
    feature_names: list,
) -> set:
    """
    Normalize a list of categorical feature names/indices to a set of indices.

    Parameters
    ----------
    categorical_features : list or None
        Feature names or integer indices.
    feature_names : list of str
        Ordered feature names.

    Returns
    -------
    set of int
        Indices of categorical features.
    """
    if not categorical_features:
        return set()

    indices = set()
    for cf in categorical_features:
        if isinstance(cf, str):
            if cf not in feature_names:
                raise ValueError(f"Unknown categorical feature name: {cf!r}")
            indices.add(feature_names.index(cf))
        elif isinstance(cf, (int, np.integer)):
            indices.add(int(cf))
        else:
            raise TypeError(
                f"categorical_features entries must be str or int; got {type(cf)}"
            )
    return indices


def plugin_ames(
    X: np.ndarray,
    predict_fn: Callable,
    feature_names: Optional[list] = None,
    categorical_features: Optional[list] = None,
    h: Union[float, str, np.ndarray] = 'adaptive',
    trim: bool = True,
    bounds: Optional[tuple] = None,
) -> dict:
    """
    Plug-in window AMEs for all features.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (n_obs, n_features).
    predict_fn : Callable
        Prediction function.
    feature_names : list, optional
        Defaults to ['x0', 'x1', ...].
    categorical_features : list, optional
        Names or indices of binary/categorical features.
    h : float, 'adaptive', or np.ndarray
        Step size specification.
    trim : bool
        Apply the trimming weight. Default True. Set False only when the
        covariate support is unbounded (e.g. Gaussian simulation designs),
        where Omega_{j,h} is all of R^d and w == 1 identically.
    bounds : tuple, optional
        (lower, upper) support bounds. If None, computed from X.

    Returns
    -------
    dict
        Feature name -> plug-in window AME.
    """
    X = np.asarray(X, dtype=float)
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]

    cat_indices = resolve_categorical(categorical_features, feature_names)
    h_values = resolve_h(X, h)
    if trim and bounds is None:
        bounds = support_bounds(X)

    results = {}
    for idx, name in enumerate(feature_names):
        is_cat = idx in cat_indices
        if is_cat or not trim:
            weights = None
        else:
            weights = trimming_weight(X, idx, h_values[idx], bounds)
        results[name] = plugin_ame(
            X, idx, predict_fn, h_values[idx], is_cat, weights
        )
    return results


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class MarginfxResult:
    """
    Container for marginfx estimation results.

    Attributes
    ----------
    estimates : dict
        Feature name -> window AME estimate.
    std_errors : dict or None
        Feature name -> standard error.
    conf_int : dict or None
        Feature name -> (lower, upper) pointwise interval.
    simultaneous_conf_int : dict or None
        Feature name -> (lower, upper) band holding simultaneously across all
        features, from the multiplier bootstrap.
    n_obs : int
        Number of observations.
    method : str
        'debiased' for the cross-fitted influence-function estimator,
        'bootstrap-diagnostic' for the refitting bootstrap.
    h : dict or None
        Feature name -> step size used. Part of the estimand, so it is
        reported alongside the estimates.
    trimmed_fraction : dict or None
        Feature name -> fraction of observations receiving trimming weight
        zero.
    n_folds : int or None
        Cross-fitting folds, for the debiased estimator.
    n_bootstrap : int or None
        Bootstrap replicates, for the diagnostic.
    alpha : float
        Significance level. Default 0.05.
    """

    def __init__(
        self,
        estimates: dict,
        std_errors: Optional[dict] = None,
        conf_int: Optional[dict] = None,
        n_obs: int = 0,
        method: str = 'debiased',
        h: Optional[dict] = None,
        trimmed_fraction: Optional[dict] = None,
        n_folds: Optional[int] = None,
        n_bootstrap: Optional[int] = None,
        alpha: float = 0.05,
        influence: Optional[dict] = None,
        simultaneous_conf_int: Optional[dict] = None,
    ):
        self.estimates = estimates
        self.std_errors = std_errors
        self.conf_int = conf_int
        self.n_obs = n_obs
        self.method = method
        self.h = h
        self.trimmed_fraction = trimmed_fraction
        self.n_folds = n_folds
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.influence = influence
        self.simultaneous_conf_int = simultaneous_conf_int

    def tidy(self) -> pd.DataFrame:
        """
        Return results as a tidy DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: term, estimate, h, std_error, statistic, p_value,
            conf_low, conf_high, trimmed. Inference columns are present only
            when standard errors were computed.
        """
        from scipy import stats

        rows = []
        for feature, est in self.estimates.items():
            row = {"term": feature, "estimate": est}

            if self.h and feature in self.h:
                row["h"] = self.h[feature]

            if self.std_errors and feature in self.std_errors:
                se = self.std_errors[feature]
                row["std_error"] = se
                stat = est / se if se > 0 else np.nan
                row["statistic"] = stat
                row["p_value"] = (
                    2.0 * (1.0 - stats.norm.cdf(abs(stat)))
                    if np.isfinite(stat) else np.nan
                )

            if self.conf_int and feature in self.conf_int:
                row["conf_low"] = self.conf_int[feature][0]
                row["conf_high"] = self.conf_int[feature][1]

            if (self.simultaneous_conf_int
                    and feature in self.simultaneous_conf_int):
                row["simul_low"] = self.simultaneous_conf_int[feature][0]
                row["simul_high"] = self.simultaneous_conf_int[feature][1]

            if self.trimmed_fraction and feature in self.trimmed_fraction:
                row["trimmed"] = self.trimmed_fraction[feature]

            rows.append(row)

        return pd.DataFrame(rows)

    def summary(self) -> None:
        """Print a formatted coefficient table to stdout."""
        df = self.tidy()
        width = 88
        print("=" * width)
        print("marginfx: Window Average Marginal Effects")
        print("=" * width)
        print(f"Observations: {self.n_obs}")

        if self.method == 'debiased':
            print(f"Estimator: debiased cross-fitted (K={self.n_folds})")
            print("Standard errors: influence function")
        elif self.method == 'bootstrap-diagnostic':
            print(f"Estimator: plug-in, refitting bootstrap "
                  f"(B={self.n_bootstrap})")
            print("WARNING: diagnostic only -- these standard errors are not")
            print("         valid for inference. See bootstrap.py.")

        if self.std_errors:
            print(f"Confidence level: {int((1 - self.alpha) * 100)}%")
        print("-" * width)
        print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        print("=" * width)

    def __repr__(self) -> str:
        return (
            f"MarginfxResult("
            f"features={len(self.estimates)}, "
            f"method={self.method!r}, "
            f"n_obs={self.n_obs})"
        )
