"""
bootstrap.py
------------
Nonparametric bootstrap for computing standard errors and confidence intervals
on average marginal effects (AMEs).

Design:
    - Fully nonparametric — no distributional assumptions
    - AMEs computed on the bootstrap sample (not the original X)
    - Warm-start from the original fitted model for computational efficiency
    - Standard errors from SD of bootstrap distribution
    - Confidence intervals from percentiles of bootstrap distribution

The engine layer provides two callables:
    fit_fn(model, X, y) -> fitted_model
        Refits the model on a new dataset, warm-starting from the original.
    predict_fn(X) -> np.ndarray
        Returns predictions of shape (n_obs,). Already bound to the model
        by the engine layer — takes only X as argument.

Step size (h):
    Default is 'adaptive', which computes h_j = max(1e-4, 0.05 * std(X[:, j]))
    per feature on the bootstrap sample, with an additional floor of 0.5 for
    integer-valued features. This ensures finite differences are meaningful
    relative to each feature's natural scale across all model types.

    Importantly, adaptive h is recomputed on each bootstrap sample — so the
    step size naturally reflects the scale of the resampled data, which is
    essentially identical to the original data for large n.

Short-circuit:
    Pass n_bootstrap=0 to compute point estimates only with no standard errors.
    This is useful for simulation studies where only AME recovery is measured.
"""

import numpy as np
from typing import Callable, Optional, List, Union
from .core import all_ames, MarginfxResult


# ---------------------------------------------------------------------------
# Single bootstrap replicate
# ---------------------------------------------------------------------------

def _bootstrap_replicate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    fit_fn: Callable,
    feature_names: Optional[List[str]],
    categorical_features: Optional[list],
    h: Union[float, str],
    rng: np.random.Generator,
) -> dict:
    """
    Run a single bootstrap replicate.

    Resamples the data, refits the model warm-starting from the original,
    and computes AMEs on the bootstrap sample.

    Parameters
    ----------
    model : fitted model object
        The original fitted model. Used as warm-start initialization.
    X : np.ndarray
        Original feature matrix, shape (n_obs, n_features).
    y : np.ndarray
        Original target vector, shape (n_obs,).
    fit_fn : Callable
        Function with signature fit_fn(model, X_boot, y_boot) -> fitted_model.
        The engine provides this. It should warm-start from model.
    feature_names : list or None
        Feature names passed through to all_ames().
    categorical_features : list or None
        Categorical feature indices or names passed through to all_ames()
    h : float or 'adaptive'
        Step size for finite differences. Passed through to all_ames().
        When 'adaptive', h is computed from the bootstrap sample itself.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    dict
        Feature name -> AME estimate for this replicate.
    """
    n = X.shape[0]

    # Resample with replacement
    indices = rng.integers(0, n, size=n)
    X_boot = X[indices]
    y_boot = y[indices]

    # Refit model warm-starting from original
    fitted = fit_fn(model, X_boot, y_boot)

    # Build a predict_fn bound to the newly fitted replicate model.
    # Autodetect the right prediction method from the fitted model.
    if hasattr(fitted, 'predict_proba'):
        def replicate_predict_fn(X_input):
            return fitted.predict_proba(X_input)[:, 1]
    elif hasattr(fitted, 'predict'):
        def replicate_predict_fn(X_input):
            return fitted.predict(X_input)
    else:
        # TF/PyTorch: model is directly callable
        def replicate_predict_fn(X_input):
            return fitted(X_input)

    # Compute AMEs on the bootstrap sample
    # When h='adaptive', step sizes are computed from X_boot itself
    return all_ames(
        X=X_boot,
        predict_fn=replicate_predict_fn,
        feature_names=feature_names,
        categorical_features=categorical_features,
        h=h,
    )


# ---------------------------------------------------------------------------
# Main bootstrap function
# ---------------------------------------------------------------------------

def bootstrap_ames(
    model,
    X: np.ndarray,
    y: np.ndarray,
    fit_fn: Callable,
    predict_fn: Callable,
    feature_names: Optional[List[str]] = None,
    categorical_features: Optional[list] = None,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    h: Union[float, str] = 'adaptive',
    seed: Optional[int] = None,
    verbose: bool = True,
) -> MarginfxResult:
    """
    Compute AMEs with bootstrap standard errors and percentile confidence intervals.

    Runs n_bootstrap replicates of:
        1. Resample data with replacement
        2. Warm-start refit from original model
        3. Compute AMEs on bootstrap sample

    Then aggregates across replicates to produce:
        - Point estimates (AMEs on full original dataset)
        - Standard errors (SD of bootstrap distribution)
        - Confidence intervals (percentiles of bootstrap distribution)

    Parameters
    ----------
    model : fitted model object
        Original fitted model. Must be compatible with fit_fn and predict_fn.
    X : np.ndarray
        Feature matrix, shape (n_obs, n_features).
    y : np.ndarray
        Target vector, shape (n_obs,).
    fit_fn : Callable
        Engine-provided function: fit_fn(model, X_boot, y_boot) -> fitted_model.
        Should warm-start from model.
    predict_fn : Callable
        Engine-provided function: predict_fn(X) -> np.ndarray.
        Already bound to the original model — takes only X as argument.
    feature_names : list, optional
        Feature names. Defaults to ['x0', 'x1', ...].
    categorical_features : list, optional
        Indices or names of categorical/binary features.
    n_bootstrap : int
        Number of bootstrap replicates. Default 200.
        Pass 0 to compute point estimates only with no standard errors.
        Useful for simulation studies measuring AME recovery.
    alpha : float
        Significance level. Default 0.05 gives 95% CIs.
    h : float or 'adaptive'
        Step size for finite differences. Default 'adaptive' computes
        h_j = max(1e-4, 0.05 * std(X[:, j])) per feature, with an
        additional floor of 0.5 for integer-valued features.
        Pass a float to use a fixed step size for all features.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        If True, prints progress every 10% of replicates.

    Returns
    -------
    MarginfxResult
        Contains point estimates, and optionally standard errors and
        confidence intervals if n_bootstrap > 0.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n_obs, n_features = X.shape

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]

    rng = np.random.default_rng(seed)

    # --- Point estimates on full dataset ---
    # predict_fn is already bound to the model — just pass X
    point_estimates = all_ames(
        X=X,
        predict_fn=predict_fn,
        feature_names=feature_names,
        categorical_features=categorical_features,
        h=h,
    )

    # --- Short-circuit if no bootstrap requested ---
    if n_bootstrap == 0:
        return MarginfxResult(
            estimates=point_estimates,
            std_errors=None,
            conf_int=None,
            n_obs=n_obs,
            n_bootstrap=0,
            alpha=alpha,
        )

    # --- Bootstrap replicates ---
    # Store as dict of feature -> list of B AME estimates
    bootstrap_distributions = {name: [] for name in feature_names}

    log_interval = max(1, n_bootstrap // 10)

    for b in range(n_bootstrap):
        if verbose and (b + 1) % log_interval == 0:
            print(f"  Bootstrap replicate {b + 1}/{n_bootstrap}...")

        replicate_ames = _bootstrap_replicate(
            model=model,
            X=X,
            y=y,
            fit_fn=fit_fn,
            feature_names=feature_names,
            categorical_features=categorical_features,
            h=h,
            rng=rng,
        )

        for name in feature_names:
            bootstrap_distributions[name].append(replicate_ames[name])

    # --- Aggregate bootstrap distributions ---
    std_errors = {}
    conf_int = {}
    lower_pct = (alpha / 2) * 100
    upper_pct = (1 - alpha / 2) * 100

    for name in feature_names:
        boot_dist = np.array(bootstrap_distributions[name])
        std_errors[name] = float(np.std(boot_dist, ddof=1))
        conf_int[name] = (
            float(np.percentile(boot_dist, lower_pct)),
            float(np.percentile(boot_dist, upper_pct)),
        )

    return MarginfxResult(
        estimates=point_estimates,
        std_errors=std_errors,
        conf_int=conf_int,
        n_obs=n_obs,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )
