"""
dml.py
------
Debiased, cross-fitted estimation of window average marginal effects.

This is the estimator the paper uses for inference. For each feature j it
computes

    theta_hat = (1/n) sum_k sum_{i in I_k} [ w(x_i) D_h f^(-k)(x_i)
                                             + alpha^(-k)(x_i) (y_i - f^(-k)(x_i)) ]

where f^(-k) and alpha^(-k) are the learner and the Riesz representer fitted on
the observations outside fold k.

Why the correction term
-----------------------
The first term alone is the plug-in average, which inherits the learner's
regularization bias. The moment bias of the corrected score is exactly

    E[(alpha_h - alpha_tilde)(f_tilde - f)]

-- a product of the two nuisance errors, with no linearization remainder. So an
error in the fitted learner harms the estimate only to the extent that the
representer is also misestimated. It is this multiplication of errors, not any
smoothness of the learner, that buys the parametric rate. Each held-out
residual, weighted by the representer, testifies to how the learner is locally
mis-calibrated where that mis-calibration matters for the marginal effect.

Cross-fitting decouples the randomness of the nuisance estimates from the
observations at which the score is evaluated, so no entropy or Donsker
condition on the learner's function class is required -- which matters here,
since boosted trees and deep networks are exactly the classes for which such
conditions fail.

Standard errors
---------------
The summands, centered at theta_hat, are estimates of the influence function
values. The standard error is their sample standard deviation over sqrt(n):

    V_hat = (1/n) sum_i (psi_i - theta_hat)^2,   se = sqrt(V_hat / n)

No resampling is involved. The whole coefficient table costs K model fits,
in place of the hundreds a refitting bootstrap requires.

For simultaneous bands across features, a multiplier bootstrap perturbs the
estimated influence function values with i.i.d. mean-zero weights, at no
additional model fits.

Double robustness
-----------------
Setting alpha_tilde = alpha_h makes the moment bias vanish identically for
every f_tilde. When the representer is known in closed form -- as in a
simulation design with known covariate density, via riesz.KnownRiesz --
inference requires no consistency from the learner at all; its failures are
paid in variance, never in location.
"""

import numpy as np
from typing import Callable, List, Optional, Union

from .core import (
    MarginfxResult,
    contrast,
    resolve_categorical,
    resolve_h,
    support_bounds,
    trimming_weight,
    window_difference,
)
from .learner import Learner
from .riesz import PropensityRiesz, SieveRiesz


# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------

def _kfold_indices(n: int, n_folds: int, rng) -> list:
    """
    Build shuffled K-fold train/test index pairs.

    Parameters
    ----------
    n : int
        Number of observations.
    n_folds : int
        Number of folds, at least 2.
    rng : np.random.Generator
        Source of randomness for the shuffle.

    Returns
    -------
    list of tuple
        (train_idx, test_idx) per fold.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2; got {n_folds}")
    if n_folds > n:
        raise ValueError(
            f"n_folds ({n_folds}) cannot exceed the number of observations ({n})"
        )

    order = rng.permutation(n)
    splits = np.array_split(order, n_folds)
    folds = []
    for k in range(n_folds):
        test_idx = splits[k]
        train_idx = np.concatenate([splits[m] for m in range(n_folds) if m != k])
        folds.append((train_idx, test_idx))
    return folds


# ---------------------------------------------------------------------------
# Representer resolution
# ---------------------------------------------------------------------------

def _make_representer(
    riesz,
    name: str,
    feature_idx: int,
    h: float,
    is_categorical: bool,
    sieve_degree: int,
    sieve_ridge: float,
    alpha_bound: Optional[float],
):
    """
    Build a fresh representer for one (fold, feature) pair.

    A fresh instance is required per fold: SieveRiesz stores fitted
    coefficients, so reusing one instance across folds would leak the last
    fold's fit.

    Parameters
    ----------
    riesz : None, dict, or Callable
        None uses the defaults -- SieveRiesz for continuous features,
        PropensityRiesz for categorical ones. A dict maps feature name to a
        representer instance (deep-copied per fold). A callable is a factory
        invoked as riesz(feature_idx, h, is_categorical).
    name : str
        Feature name.
    feature_idx : int
        Feature index.
    h : float
        Step size for this feature.
    is_categorical : bool
        Whether the feature is binary/categorical.
    sieve_degree, sieve_ridge, alpha_bound
        Defaults forwarded to SieveRiesz.

    Returns
    -------
    object
        An object exposing fit(X, feature_idx, h, weights) and predict(X).
    """
    if riesz is None:
        if is_categorical:
            return PropensityRiesz()
        return SieveRiesz(
            degree=sieve_degree, ridge=sieve_ridge, alpha_bound=alpha_bound
        )

    if isinstance(riesz, dict):
        import copy
        if name in riesz:
            return copy.deepcopy(riesz[name])
        return _make_representer(
            None, name, feature_idx, h, is_categorical,
            sieve_degree, sieve_ridge, alpha_bound,
        )

    if callable(riesz):
        return riesz(feature_idx, h, is_categorical)

    raise TypeError(
        "riesz must be None, a dict of feature name -> representer, or a "
        f"factory callable; got {type(riesz).__name__}."
    )


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

def crossfit_ames(
    learner,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    categorical_features: Optional[list] = None,
    h: Union[float, str, np.ndarray] = 'adaptive',
    trim: bool = True,
    n_folds: int = 5,
    riesz=None,
    sieve_degree: int = 2,
    sieve_ridge: float = 1e-6,
    alpha_bound: Optional[float] = None,
    alpha: float = 0.05,
    n_multiplier: int = 0,
    seed: Optional[int] = None,
    verbose: bool = True,
    n_epochs: int = 10,
    batch_size: int = 32,
    optimizer_fn: Optional[Callable] = None,
    loss_fn=None,
) -> MarginfxResult:
    """
    Debiased cross-fitted window AMEs with influence-function standard errors.

    Parameters
    ----------
    learner : estimator or callable
        An unfitted scikit-learn compatible estimator, or a zero-argument
        callable returning a fresh model. It is fitted K times, once per
        training fold. A pre-fitted model cannot be used: cross-fitting
        requires that the model never see the fold it is evaluated on.
    X : np.ndarray
        Feature matrix, shape (n_obs, n_features).
    y : np.ndarray
        Target vector, shape (n_obs,).
    feature_names : list of str, optional
        Defaults to ['x0', 'x1', ...].
    categorical_features : list, optional
        Names or indices of binary features. These use the contrast and an
        inverse propensity representer.
    h : float, 'adaptive', or np.ndarray
        Step size. Default 'adaptive' gives h_j = max(1e-4, 0.05 * std_j).
        h defines the estimand, so it is computed once on the full sample and
        held fixed across folds.
    trim : bool
        Apply the trimming weight. Default True. Set False only when the
        support is unbounded, e.g. Gaussian simulation designs.
    n_folds : int
        Cross-fitting folds K. Default 5.
    riesz : None, dict, or Callable
        Representer specification. None uses Riesz regression over a
        polynomial sieve for continuous features and inverse propensity
        weights for categorical ones. Pass a factory
        riesz(feature_idx, h, is_categorical) to supply a known representer
        -- see riesz.gaussian_window_riesz.
    sieve_degree : int
        Polynomial degree for the default sieve. Default 2.
    sieve_ridge : float
        Ridge penalty for the default sieve. Default 1e-6.
    alpha_bound : float, optional
        Truncation bound for the estimated representer.
    alpha : float
        Significance level. Default 0.05.
    n_multiplier : int
        Multiplier bootstrap draws for simultaneous bands across features.
        Default 0 (skip). Costs no additional model fits.
    seed : int, optional
        Random seed for fold assignment and the multiplier bootstrap.
    verbose : bool
        Print per-fold progress.
    n_epochs, batch_size, optimizer_fn, loss_fn
        Forwarded to Learner for Keras and PyTorch specifications.

    Returns
    -------
    MarginfxResult
        With method='debiased'. The per-observation influence values are
        retained on `.influence` for diagnostics.
    """
    from scipy import stats

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape

    if y.shape[0] != n:
        raise ValueError(
            f"X has {n} rows but y has {y.shape[0]}"
        )

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(d)]
    if len(feature_names) != d:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but X has {d} columns"
        )

    cat_indices = resolve_categorical(categorical_features, feature_names)

    # h and the trimming weight define the estimand: compute once on the full
    # sample, then hold fixed. Letting either drift with the fold would mean
    # each fold targets a slightly different functional.
    h_values = resolve_h(X, h)
    bounds = support_bounds(X)

    W = np.ones((n, d), dtype=float)
    if trim:
        for j in range(d):
            if j not in cat_indices:
                W[:, j] = trimming_weight(X, j, h_values[j], bounds)

    fit_engine = Learner(
        learner,
        n_epochs=n_epochs,
        batch_size=batch_size,
        optimizer_fn=optimizer_fn,
        loss_fn=loss_fn,
    )

    rng = np.random.default_rng(seed)
    folds = _kfold_indices(n, n_folds, rng)

    psi = np.full((n, d), np.nan, dtype=float)

    for k, (train_idx, test_idx) in enumerate(folds):
        if verbose:
            print(f"  Fold {k + 1}/{n_folds} "
                  f"(train={len(train_idx)}, test={len(test_idx)})...")

        predict_fn = fit_engine.fit(X[train_idx], y[train_idx])

        X_test = X[test_idx]
        resid = y[test_idx] - predict_fn(X_test)

        for j, name in enumerate(feature_names):
            is_cat = j in cat_indices

            representer = _make_representer(
                riesz, name, j, h_values[j], is_cat,
                sieve_degree, sieve_ridge, alpha_bound,
            )
            representer.fit(X[train_idx], j, h_values[j], W[train_idx, j])
            alpha_test = representer.predict(X_test)

            if is_cat:
                base = contrast(X_test, j, predict_fn)
            else:
                base = W[test_idx, j] * window_difference(
                    X_test, j, predict_fn, h_values[j]
                )

            psi[test_idx, j] = base + alpha_test * resid

    if np.isnan(psi).any():
        raise RuntimeError(
            "Cross-fitting left some observations unscored; this is a bug."
        )

    theta = psi.mean(axis=0)
    variance = ((psi - theta) ** 2).mean(axis=0)
    se = np.sqrt(variance / n)

    z = stats.norm.ppf(1.0 - alpha / 2.0)

    estimates = {}
    std_errors = {}
    conf_int = {}
    h_report = {}
    trimmed_fraction = {}

    for j, name in enumerate(feature_names):
        estimates[name] = float(theta[j])
        std_errors[name] = float(se[j])
        conf_int[name] = (
            float(theta[j] - z * se[j]),
            float(theta[j] + z * se[j]),
        )
        h_report[name] = (
            float('nan') if j in cat_indices else float(h_values[j])
        )
        trimmed_fraction[name] = (
            0.0 if j in cat_indices else float(1.0 - W[:, j].mean())
        )

    simultaneous = None
    if n_multiplier > 0:
        simultaneous = _multiplier_bands(
            psi, theta, se, feature_names, alpha, n_multiplier, rng
        )

    return MarginfxResult(
        estimates=estimates,
        std_errors=std_errors,
        conf_int=conf_int,
        n_obs=n,
        method='debiased',
        h=h_report,
        trimmed_fraction=trimmed_fraction,
        n_folds=n_folds,
        alpha=alpha,
        influence={name: psi[:, j] for j, name in enumerate(feature_names)},
        simultaneous_conf_int=simultaneous,
    )


# ---------------------------------------------------------------------------
# Multiplier bootstrap
# ---------------------------------------------------------------------------

def _multiplier_bands(
    psi: np.ndarray,
    theta: np.ndarray,
    se: np.ndarray,
    feature_names: List[str],
    alpha: float,
    n_draws: int,
    rng,
) -> dict:
    """
    Simultaneous confidence bands via the multiplier bootstrap.

    Perturbs the centered influence values with i.i.d. standard normal
    weights, forming

        theta_b = theta_hat + (1/n) sum_i xi_i (psi_i - theta_hat)

    and takes the (1 - alpha) quantile of the max-t statistic across features
    as the critical value. Requires no additional model fits.

    Parameters
    ----------
    psi : np.ndarray
        Influence values, shape (n_obs, n_features).
    theta : np.ndarray
        Point estimates, shape (n_features,).
    se : np.ndarray
        Standard errors, shape (n_features,).
    feature_names : list of str
        Feature names.
    alpha : float
        Significance level.
    n_draws : int
        Number of multiplier draws.
    rng : np.random.Generator
        Randomness source.

    Returns
    -------
    dict
        Feature name -> (lower, upper) simultaneous band.
    """
    n = psi.shape[0]
    centered = psi - theta

    safe_se = np.where(se > 0, se, np.inf)

    max_t = np.empty(n_draws, dtype=float)
    for b in range(n_draws):
        xi = rng.standard_normal(n)
        perturbed = (xi @ centered) / n
        max_t[b] = np.max(np.abs(perturbed) / safe_se)

    crit = float(np.quantile(max_t, 1.0 - alpha))

    return {
        name: (float(theta[j] - crit * se[j]), float(theta[j] + crit * se[j]))
        for j, name in enumerate(feature_names)
    }
