"""
bootstrap.py
------------
Refitting bootstrap for the plug-in window AMEs.

Consistent for the variability of the plug-in average around its own
expectation under the learner, and blind to the distance between that
expectation and the true window effect. Valid where the learner's
regularization bias is small relative to the standard error; silently
invalid where it is not, increasingly so as n grows. See fit() for the
debiased estimator, which removes that term instead of assuming it away.

What this measures
------------------
Resampling (x_i, y_i) pairs jointly, refitting the learner on each replicate,
and recomputing the plug-in window AME measures how sensitive the reported
effects are to the joint variability of resampling, refitting, and
hyperparameter selection. Large dispersion is a useful warning that the fitted
model is unstable.

What this does NOT measure
--------------------------
It is not a substitute for the influence-function standard errors in dml.py,
for two reasons the paper makes precise.

1.  It recenters at the fitted model, and is therefore blind to the
    regularization bias that the orthogonal correction removes. Every replicate
    recenters at a similarly regularized fit, so resampling cannot see the bias
    term E[alpha_h (f_hat - f)]. The intervals shrink at the right rate around
    a displaced point. In the paper's simulations this produces random forest
    coverage collapsing to roughly 0.10-0.13 at n = 5,000 -- the interval ends
    up about the width of the bias itself. The failure is not specific to
    forests but to any learner whose L^2 bias shrinks more slowly than its
    intervals.

2.  Because each replicate is evaluated at the original sample D, it omits the
    component of sampling variance that comes from averaging a heterogeneous
    effect over a finite sample of covariate values.

Evaluation point
----------------
Replicate effects are computed with f_hat^(b) evaluated at the ORIGINAL sample
D, not at the bootstrap sample D^(b). This follows standard practice for smooth
functionals of the empirical distribution and makes the replicate depend on the
draw only through f_hat^(b). It is also the choice the paper analyzes.

Consequently h and the trimming weight are computed once on the original
sample and held fixed -- they define the estimand, so they must not be
recomputed per replicate.
"""

import numpy as np
from typing import Callable, List, Optional, Union

from .core import (
    MarginfxResult,
    plugin_ames,
    resolve_categorical,
    resolve_h,
    support_bounds,
    trimming_weight,
)


# ---------------------------------------------------------------------------
# Single replicate
# ---------------------------------------------------------------------------

def _bootstrap_replicate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    fit_fn: Callable,
    feature_names: List[str],
    categorical_features: Optional[list],
    h_values: np.ndarray,
    trim: bool,
    bounds: tuple,
    rng: np.random.Generator,
) -> dict:
    """
    Run a single bootstrap replicate.

    Resamples (x_i, y_i) pairs jointly with replacement, refits the model
    warm-starting from the original, and recomputes the plug-in window AMEs
    at the ORIGINAL sample X.

    Parameters
    ----------
    model : fitted model
        Original fitted model, used as the warm start.
    X : np.ndarray
        Original feature matrix, shape (n_obs, n_features).
    y : np.ndarray
        Original target vector, shape (n_obs,).
    fit_fn : Callable
        fit_fn(model, X_boot, y_boot) -> fitted_model, from the engine layer.
    feature_names : list of str
        Feature names.
    categorical_features : list or None
        Categorical feature names or indices.
    h_values : np.ndarray
        Per-feature step sizes, fixed from the original sample.
    trim : bool
        Whether to apply the trimming weight.
    bounds : tuple
        (lower, upper) support bounds from the original sample.
    rng : np.random.Generator
        Randomness source.

    Returns
    -------
    dict
        Feature name -> replicate plug-in window AME.
    """
    n = X.shape[0]

    indices = rng.integers(0, n, size=n)
    fitted = fit_fn(model, X[indices], y[indices])

    if hasattr(fitted, 'predict_proba'):
        def replicate_predict_fn(X_input):
            proba = fitted.predict_proba(X_input)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba
    elif hasattr(fitted, 'predict'):
        def replicate_predict_fn(X_input):
            return fitted.predict(X_input)
    else:
        def replicate_predict_fn(X_input):
            return fitted(X_input)

    # Evaluated at the ORIGINAL sample X, with the estimand's fixed h and w.
    return plugin_ames(
        X=X,
        predict_fn=replicate_predict_fn,
        feature_names=feature_names,
        categorical_features=categorical_features,
        h=h_values,
        trim=trim,
        bounds=bounds,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def bootstrap_diagnostic(
    model,
    X: np.ndarray,
    y: np.ndarray,
    fit_fn: Callable,
    predict_fn: Callable,
    feature_names: Optional[List[str]] = None,
    categorical_features: Optional[list] = None,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    h: Union[float, str, np.ndarray] = 'adaptive',
    trim: bool = True,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> MarginfxResult:
    """
    Plug-in window AMEs with refitting-bootstrap dispersion, as a diagnostic.

    The returned standard errors are NOT valid for inference. Use
    marginfx.fit() for that. See the module docstring for why.

    Parameters
    ----------
    model : fitted model
        Original fitted model, compatible with fit_fn and predict_fn.
    X : np.ndarray
        Feature matrix, shape (n_obs, n_features).
    y : np.ndarray
        Target vector, shape (n_obs,).
    fit_fn : Callable
        fit_fn(model, X_boot, y_boot) -> fitted_model.
    predict_fn : Callable
        predict_fn(X) -> np.ndarray, already bound to `model`.
    feature_names : list of str, optional
        Defaults to ['x0', 'x1', ...].
    categorical_features : list, optional
        Names or indices of binary features.
    n_bootstrap : int
        Replicates. Default 200. Pass 0 for point estimates only.
    alpha : float
        Significance level for the percentile intervals. Default 0.05.
    h : float, 'adaptive', or np.ndarray
        Step size. Computed once on the original sample and held fixed.
    trim : bool
        Apply the trimming weight. Default True.
    seed : int, optional
        Random seed.
    verbose : bool
        Print progress every 10% of replicates.

    Returns
    -------
    MarginfxResult
        With method='bootstrap-diagnostic'.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_obs, n_features = X.shape

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]

    cat_indices = resolve_categorical(categorical_features, feature_names)
    h_values = resolve_h(X, h)
    bounds = support_bounds(X)

    h_report = {}
    trimmed_fraction = {}
    for j, name in enumerate(feature_names):
        if j in cat_indices:
            h_report[name] = float('nan')
            trimmed_fraction[name] = 0.0
        else:
            h_report[name] = float(h_values[j])
            w = (trimming_weight(X, j, h_values[j], bounds)
                 if trim else np.ones(n_obs))
            trimmed_fraction[name] = float(1.0 - w.mean())

    point_estimates = plugin_ames(
        X=X,
        predict_fn=predict_fn,
        feature_names=feature_names,
        categorical_features=categorical_features,
        h=h_values,
        trim=trim,
        bounds=bounds,
    )

    if n_bootstrap == 0:
        return MarginfxResult(
            estimates=point_estimates,
            n_obs=n_obs,
            method='bootstrap-diagnostic',
            h=h_report,
            trimmed_fraction=trimmed_fraction,
            n_bootstrap=0,
            alpha=alpha,
        )

    rng = np.random.default_rng(seed)
    distributions = {name: [] for name in feature_names}
    log_interval = max(1, n_bootstrap // 10)

    for b in range(n_bootstrap):
        if verbose and (b + 1) % log_interval == 0:
            print(f"  Bootstrap replicate {b + 1}/{n_bootstrap}...")

        replicate = _bootstrap_replicate(
            model=model,
            X=X,
            y=y,
            fit_fn=fit_fn,
            feature_names=feature_names,
            categorical_features=categorical_features,
            h_values=h_values,
            trim=trim,
            bounds=bounds,
            rng=rng,
        )
        for name in feature_names:
            distributions[name].append(replicate[name])

    std_errors = {}
    conf_int = {}
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    for name in feature_names:
        dist = np.asarray(distributions[name], dtype=float)
        std_errors[name] = float(np.std(dist, ddof=1))
        conf_int[name] = (
            float(np.percentile(dist, lower_pct)),
            float(np.percentile(dist, upper_pct)),
        )

    return MarginfxResult(
        estimates=point_estimates,
        std_errors=std_errors,
        conf_int=conf_int,
        n_obs=n_obs,
        method='bootstrap-diagnostic',
        h=h_report,
        trimmed_fraction=trimmed_fraction,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )
