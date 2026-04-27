"""
marginfx
--------
Average marginal effects and bootstrap standard errors for any machine
learning model. Get OLS-style interpretability from scikit-learn, XGBoost,
LightGBM, TensorFlow, and PyTorch.

Basic usage
-----------
>>> import marginfx as mfx
>>>
>>> # Works with any model
>>> result = mfx.fit(model, X, y)
>>> result.summary()
>>> result.tidy()

Supported model families
------------------------
- scikit-learn (RandomForest, GradientBoosting, LogisticRegression, etc.)
- XGBoost (XGBClassifier, XGBRegressor)
- LightGBM (LGBMClassifier, LGBMRegressor)
- TensorFlow / Keras (tf.keras.Model)
- PyTorch (torch.nn.Module)

Model type is detected automatically. No need to specify the engine.

For TensorFlow and PyTorch models, exact gradients are used instead of
finite differences — faster and more accurate.
"""

import numpy as np
from typing import Callable, List, Optional, Union

from .core import MarginfxResult, all_ames
from .bootstrap import bootstrap_ames


# ---------------------------------------------------------------------------
# Model type detection
# ---------------------------------------------------------------------------

def _detect_engine(model) -> str:
    """
    Detect model type and return engine name.

    Detection order:
        1. PyTorch — torch.nn.Module
        2. TensorFlow — tf.keras.Model
        3. Sklearn — anything with .predict()
        4. Unknown — raise helpful error

    Uses broad Exception catch for PyTorch and TensorFlow to handle
    broken installations gracefully (e.g. DLL errors on Windows).

    Parameters
    ----------
    model : any
        A fitted model object.

    Returns
    -------
    str
        One of 'pytorch', 'tensorflow', 'sklearn'.

    Raises
    ------
    TypeError
        If model type is not recognized.
    """
    # PyTorch
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            return 'pytorch'
    except Exception:
        pass

    # TensorFlow / Keras
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            return 'tensorflow'
    except Exception:
        pass

    # Sklearn-compatible — anything with .predict()
    if hasattr(model, 'predict'):
        return 'sklearn'

    raise TypeError(
        f"marginfx could not detect the model type for {type(model).__name__}. "
        f"Model must be a torch.nn.Module, tf.keras.Model, or any object "
        f"with a .predict() method (scikit-learn, XGBoost, LightGBM, etc.)."
    )


# ---------------------------------------------------------------------------
# Engine loader
# ---------------------------------------------------------------------------

def _load_engine(model, engine_name: str, **kwargs):
    """
    Load the appropriate engine and return (predict_fn, fit_fn, gradient_ame_fn).

    gradient_ame_fn is None for sklearn models — they use finite differences.

    Parameters
    ----------
    model : fitted model
    engine_name : str
        One of 'pytorch', 'tensorflow', 'sklearn'.
    **kwargs
        Engine-specific keyword arguments passed through.

    Returns
    -------
    tuple
        (predict_fn, fit_fn, gradient_ame_fn or None)
    """
    if engine_name == 'pytorch':
        from .engines.pytorch import get_engine
        optimizer_fn = kwargs.get('optimizer_fn', None)
        loss_fn = kwargs.get('loss_fn', None)
        n_epochs = kwargs.get('n_epochs', 10)
        batch_size = kwargs.get('batch_size', 32)

        engine_args = {}
        if optimizer_fn is not None:
            engine_args['optimizer_fn'] = optimizer_fn
        if loss_fn is not None:
            engine_args['loss_fn'] = loss_fn

        predict_fn, fit_fn, gradient_ame_fn = get_engine(
            model,
            n_epochs=n_epochs,
            batch_size=batch_size,
            **engine_args,
        )
        return predict_fn, fit_fn, gradient_ame_fn

    elif engine_name == 'tensorflow':
        from .engines.tensorflow import get_engine
        n_epochs = kwargs.get('n_epochs', 10)
        batch_size = kwargs.get('batch_size', 32)
        predict_fn, fit_fn, gradient_ame_fn = get_engine(
            model,
            n_epochs=n_epochs,
            batch_size=batch_size,
        )
        return predict_fn, fit_fn, gradient_ame_fn

    elif engine_name == 'sklearn':
        from .engines.sklearn import get_engine
        predict_fn, fit_fn = get_engine(model)
        return predict_fn, fit_fn, None

    else:
        raise ValueError(f"Unknown engine: {engine_name}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit(
    model,
    X: np.ndarray,
    y: np.ndarray,
    # Data options
    feature_names: Optional[List[str]] = None,
    categorical_features: Optional[list] = None,
    # Bootstrap options
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    verbose: bool = True,
    # Finite difference options
    h: Union[float, str] = 'adaptive',
    # TensorFlow / PyTorch options
    n_epochs: int = 10,
    batch_size: int = 32,
    # PyTorch only
    optimizer_fn: Optional[Callable] = None,
    loss_fn=None,
) -> MarginfxResult:
    """
    Compute average marginal effects with bootstrap standard errors and
    confidence intervals for any machine learning model.

    Model type is detected automatically. For TensorFlow and PyTorch models,
    exact gradients replace finite differences automatically.

    Parameters
    ----------
    model : fitted model
        Any fitted model. Supported types:
            - scikit-learn models (RandomForest, LogisticRegression, etc.)
            - XGBoost (XGBClassifier, XGBRegressor)
            - LightGBM (LGBMClassifier, LGBMRegressor)
            - TensorFlow / Keras (tf.keras.Model)
            - PyTorch (torch.nn.Module)
    X : np.ndarray
        Feature matrix, shape (n_obs, n_features).
    y : np.ndarray
        Target vector, shape (n_obs,).
    feature_names : list of str, optional
        Names for each feature column. Defaults to ['x0', 'x1', ...].
        If X is a pandas DataFrame, column names are used automatically.
    categorical_features : list, optional
        Indices or names of binary/categorical features. These use first
        differences (0 -> 1) instead of derivatives.
    n_bootstrap : int
        Number of bootstrap replicates. Default 200.
        More replicates = more accurate SEs but slower. 200 is reasonable
        for most applications.
    alpha : float
        Significance level for confidence intervals. Default 0.05 (95% CIs).
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        Print bootstrap progress. Default True.
    h : float or 'adaptive'
        Step size for finite difference approximation. Default 'adaptive'
        computes h_j = max(1e-4, 0.01 * std(X[:, j])) per feature.
        Pass a float to use a fixed step size for all features.
        Only used for sklearn models — TensorFlow and PyTorch use exact
        gradients regardless of h.
    n_epochs : int
        Epochs for bootstrap warm-start refit. TensorFlow and PyTorch only.
        Default 10.
    batch_size : int
        Batch size for bootstrap warm-start refit. TensorFlow and PyTorch
        only. Default 32.
    optimizer_fn : Callable, optional
        PyTorch only. Callable that takes model parameters and returns an
        optimizer. Default: Adam with lr=1e-3.
        Example: lambda params: torch.optim.SGD(params, lr=0.01)
    loss_fn : torch loss function, optional
        PyTorch only. Default: BCELoss for binary classification.
        Example: torch.nn.MSELoss() for regression.

    Returns
    -------
    MarginfxResult
        Contains:
            .estimates   -- AME point estimates (dict)
            .std_errors  -- bootstrap standard errors (dict)
            .conf_int    -- percentile confidence intervals (dict)
            .tidy()      -- tidy DataFrame (like broom::tidy() in R)
            .summary()   -- formatted summary table

    Examples
    --------
    Scikit-learn random forest:
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> import marginfx as mfx
    >>>
    >>> model = RandomForestClassifier().fit(X_train, y_train)
    >>> result = mfx.fit(model, X, y, feature_names=feature_names, seed=42)
    >>> result.summary()

    XGBoost:
    >>> import xgboost as xgb
    >>> import marginfx as mfx
    >>>
    >>> model = xgb.XGBClassifier().fit(X_train, y_train)
    >>> result = mfx.fit(model, X, y, seed=42)
    >>> result.summary()

    TensorFlow:
    >>> import tensorflow as tf
    >>> import marginfx as mfx
    >>>
    >>> model = tf.keras.models.load_model('my_model.keras')
    >>> result = mfx.fit(model, X, y, n_epochs=10, seed=42)
    >>> result.summary()

    PyTorch:
    >>> import torch
    >>> import torch.nn as nn
    >>> import marginfx as mfx
    >>>
    >>> result = mfx.fit(
    ...     model, X, y,
    ...     loss_fn=nn.BCELoss(),
    ...     optimizer_fn=lambda p: torch.optim.Adam(p, lr=1e-3),
    ...     n_epochs=10,
    ...     seed=42,
    ... )
    >>> result.summary()
    """
    # Handle pandas DataFrames gracefully
    if hasattr(X, 'columns') and feature_names is None:
        feature_names = list(X.columns)
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # Detect engine
    engine_name = _detect_engine(model)

    if verbose:
        print(f"marginfx: detected {engine_name} model")
        print(f"marginfx: running {n_bootstrap} bootstrap replicates...")

    # Load engine
    predict_fn, fit_fn, gradient_ame_fn = _load_engine(
        model,
        engine_name,
        n_epochs=n_epochs,
        batch_size=batch_size,
        optimizer_fn=optimizer_fn,
        loss_fn=loss_fn,
    )

    # Run bootstrap
    result = bootstrap_ames(
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
        seed=seed,
        verbose=verbose,
    )

    return result


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

__all__ = [
    'fit',
    'MarginfxResult',
]

__version__ = '0.1.0'
