"""
marginfx
--------
Window average marginal effects with valid inference for any supervised
learner.

Basic usage
-----------
>>> import marginfx as mfx
>>> from sklearn.ensemble import RandomForestClassifier
>>>
>>> result = mfx.fit(RandomForestClassifier(), X, y)
>>> result.summary()
>>> result.tidy()

Note that `fit` takes an UNFITTED learner. It fits the model itself, K times,
once per cross-fitting fold. This is not a convenience -- cross-fitting
requires that the model never see the observations at which its own score is
evaluated. For Keras and PyTorch, pass a zero-argument callable that returns a
fresh compiled model.

What is estimated
-----------------
The window average marginal effect at scale h,

    theta_{j,h} = E[ w(X) (f(X + h e_j) - f(X - h e_j)) / (2h) ]

which converges to the classical average marginal effect E[df/dx_j] as h -> 0.
h is part of the estimand rather than a numerical tolerance, and is reported
alongside the estimates. The trimming weight w excludes points whose shifted
evaluations would fall outside the observed support.

How inference works
-------------------
`fit` returns a debiased, cross-fitted estimator with standard errors from the
influence function -- no resampling, K model fits in total. The correction term
removes the first-order bias that regularization of the learner transmits to
the plug-in average. See dml.py.

`bootstrap_diagnostic` retains the older refitting bootstrap, but only as a
diagnostic for model instability. Its standard errors are not valid for
inference; see bootstrap.py for why.

Supported learners
------------------
- scikit-learn compatible estimators (RandomForest, GradientBoosting,
  LogisticRegression, LinearRegression, ...)
- XGBoost and LightGBM via their sklearn wrappers
- TensorFlow / Keras and PyTorch, passed as a factory callable

For binary classification the learner must expose predicted probabilities;
class labels are discrete and their finite differences are identically zero.
"""

import numpy as np
from typing import Callable, List, Optional, Union

from .core import MarginfxResult, compute_adaptive_h, plugin_ames
from .dml import crossfit_ames
from .bootstrap import bootstrap_diagnostic as _bootstrap_diagnostic
from .riesz import (
    KnownRiesz,
    PropensityRiesz,
    SieveRiesz,
    gaussian_window_riesz,
    uniform_window_riesz,
)


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------

def _prepare(X, y, feature_names):
    """Coerce DataFrame/Series inputs and infer feature names."""
    if hasattr(X, 'columns') and feature_names is None:
        feature_names = list(X.columns)
    if hasattr(X, 'values'):
        X = X.values
    if y is not None and hasattr(y, 'values'):
        y = y.values

    X = np.asarray(X, dtype=float)
    if y is not None:
        y = np.asarray(y, dtype=float)
    return X, y, feature_names


# ---------------------------------------------------------------------------
# Model type detection (diagnostic path only)
# ---------------------------------------------------------------------------

def _detect_engine(model) -> str:
    """
    Detect the framework of a fitted model.

    Returns
    -------
    str
        One of 'pytorch', 'tensorflow', 'sklearn'.
    """
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            return 'pytorch'
    except Exception:
        pass

    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            return 'tensorflow'
    except Exception:
        pass

    if hasattr(model, 'predict'):
        return 'sklearn'

    raise TypeError(
        f"marginfx could not detect the model type for {type(model).__name__}. "
        f"Model must be a torch.nn.Module, tf.keras.Model, or any object "
        f"with a .predict() method (scikit-learn, XGBoost, LightGBM, etc.)."
    )


def _load_engine(model, engine_name: str, **kwargs):
    """Load an engine and return (predict_fn, fit_fn) for the warm-start path."""
    if engine_name == 'pytorch':
        from .engines.pytorch import get_engine
        engine_args = {}
        if kwargs.get('optimizer_fn') is not None:
            engine_args['optimizer_fn'] = kwargs['optimizer_fn']
        if kwargs.get('loss_fn') is not None:
            engine_args['loss_fn'] = kwargs['loss_fn']
        return get_engine(
            model,
            n_epochs=kwargs.get('n_epochs', 10),
            batch_size=kwargs.get('batch_size', 32),
            **engine_args,
        )

    if engine_name == 'tensorflow':
        from .engines.tensorflow import get_engine
        return get_engine(
            model,
            n_epochs=kwargs.get('n_epochs', 10),
            batch_size=kwargs.get('batch_size', 32),
        )

    if engine_name == 'sklearn':
        from .engines.sklearn import get_engine
        return get_engine(model)

    raise ValueError(f"Unknown engine: {engine_name}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit(
    learner,
    X,
    y,
    feature_names: Optional[List[str]] = None,
    categorical_features: Optional[list] = None,
    h: Union[float, str, np.ndarray] = 'adaptive',
    trim: bool = True,
    bounds: Optional[tuple] = None,
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
    Debiased cross-fitted window average marginal effects.

    Parameters
    ----------
    learner : estimator or callable
        An UNFITTED scikit-learn compatible estimator, or a zero-argument
        callable returning a fresh model. Fitted K times, once per fold.
    X : np.ndarray or pd.DataFrame
        Feature matrix, shape (n_obs, n_features).
    y : np.ndarray or pd.Series
        Target vector, shape (n_obs,).
    feature_names : list of str, optional
        Inferred from DataFrame columns when available.
    categorical_features : list, optional
        Names or indices of binary features. These use the contrast
        f(x|x_j=1) - f(x|x_j=0) with an inverse propensity representer.
    h : float, 'adaptive', or np.ndarray
        Step size defining the estimand. Default 'adaptive' gives
        h_j = max(1e-4, 0.05 * std_j), floored at 0.5 for integer-valued
        features so that a count is contrasted over a whole unit. The
        resolved h is reported on the result.
    trim : bool
        Apply the trimming weight. Default True. Set False only when the
        covariate support is unbounded, e.g. Gaussian simulation designs.
    n_folds : int
        Cross-fitting folds K. Default 5.
    riesz : None, dict, or Callable
        Representer specification. None estimates it by Riesz regression.
        Pass a factory riesz(feature_idx, h, is_categorical) to supply a
        closed-form representer -- see marginfx.gaussian_window_riesz.
    sieve_degree : int
        Polynomial degree for the default sieve. Default 2.
    sieve_ridge : float
        Ridge penalty for the default sieve. Default 1e-6.
    alpha_bound : float, optional
        Truncation bound for the estimated representer.
    alpha : float
        Significance level. Default 0.05.
    n_multiplier : int
        Multiplier bootstrap draws for simultaneous bands. Default 0.
    seed : int, optional
        Random seed.
    verbose : bool
        Print progress.
    n_epochs, batch_size : int
        Training settings for Keras and PyTorch.
    optimizer_fn, loss_fn
        PyTorch training settings.

    Returns
    -------
    MarginfxResult

    Examples
    --------
    >>> import marginfx as mfx
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> result = mfx.fit(RandomForestClassifier(n_estimators=200), X, y, seed=42)
    >>> result.summary()

    Keras, via a factory:

    >>> def make_model():
    ...     m = tf.keras.Sequential([...])
    ...     m.compile(optimizer='adam', loss='binary_crossentropy')
    ...     return m
    >>> result = mfx.fit(make_model, X, y, n_epochs=20, seed=42)
    """
    X, y, feature_names = _prepare(X, y, feature_names)

    if verbose:
        print(f"marginfx: debiased cross-fitted estimator (K={n_folds})")

    return crossfit_ames(
        learner=learner,
        X=X,
        y=y,
        feature_names=feature_names,
        categorical_features=categorical_features,
        h=h,
        trim=trim,
        bounds=bounds,
        n_folds=n_folds,
        riesz=riesz,
        sieve_degree=sieve_degree,
        sieve_ridge=sieve_ridge,
        alpha_bound=alpha_bound,
        alpha=alpha,
        n_multiplier=n_multiplier,
        seed=seed,
        verbose=verbose,
        n_epochs=n_epochs,
        batch_size=batch_size,
        optimizer_fn=optimizer_fn,
        loss_fn=loss_fn,
    )


def bootstrap_diagnostic(
    model,
    X,
    y,
    feature_names: Optional[List[str]] = None,
    categorical_features: Optional[list] = None,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    h: Union[float, str, np.ndarray] = 'adaptive',
    trim: bool = True,
    seed: Optional[int] = None,
    verbose: bool = True,
    n_epochs: int = 10,
    batch_size: int = 32,
    optimizer_fn: Optional[Callable] = None,
    loss_fn=None,
) -> MarginfxResult:
    """
    Refitting-bootstrap dispersion of the plug-in window AMEs -- a DIAGNOSTIC.

    The standard errors returned here are not valid for inference. They are
    blind to the learner's regularization bias, which is exactly the term the
    debiased estimator in `fit` is built to remove. Use this to detect model
    instability, and `fit` to do inference.

    Unlike `fit`, this takes an ALREADY FITTED model, and refits it on each
    replicate warm-started from that fit.

    Parameters
    ----------
    model : fitted model
        A fitted scikit-learn compatible model, tf.keras.Model, or
        torch.nn.Module.
    X : np.ndarray or pd.DataFrame
        Feature matrix.
    y : np.ndarray or pd.Series
        Target vector.
    feature_names : list of str, optional
        Inferred from DataFrame columns when available.
    categorical_features : list, optional
        Names or indices of binary features.
    n_bootstrap : int
        Replicates. Default 200. Pass 0 for point estimates only.
    alpha : float
        Significance level. Default 0.05.
    h : float, 'adaptive', or np.ndarray
        Step size defining the estimand.
    trim : bool
        Apply the trimming weight. Default True.
    seed : int, optional
        Random seed.
    verbose : bool
        Print progress.
    n_epochs, batch_size, optimizer_fn, loss_fn
        Framework-specific refit settings.

    Returns
    -------
    MarginfxResult
        With method='bootstrap-diagnostic'.
    """
    X, y, feature_names = _prepare(X, y, feature_names)

    engine_name = _detect_engine(model)
    if verbose:
        print(f"marginfx: detected {engine_name} model")
        print(f"marginfx: {n_bootstrap} bootstrap replicates (DIAGNOSTIC ONLY)")

    predict_fn, fit_fn = _load_engine(
        model,
        engine_name,
        n_epochs=n_epochs,
        batch_size=batch_size,
        optimizer_fn=optimizer_fn,
        loss_fn=loss_fn,
    )

    return _bootstrap_diagnostic(
        model=model,
        X=X,
        y=y,
        fit_fn=fit_fn,
        predict_fn=predict_fn,
        feature_names=feature_names,
        categorical_features=categorical_features,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        h=h,
        trim=trim,
        seed=seed,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

__all__ = [
    'fit',
    'bootstrap_diagnostic',
    'MarginfxResult',
    'plugin_ames',
    'compute_adaptive_h',
    'SieveRiesz',
    'KnownRiesz',
    'PropensityRiesz',
    'gaussian_window_riesz',
    'uniform_window_riesz',
]

__version__ = '0.3.0'
