"""
learner.py
----------
Cold-fit adapter used by cross-fitting.

Cross-fitting needs the learner fitted from scratch on each training fold --
K independent fits, with no information crossing between folds. That is a
different requirement from the warm-start refit in engines/, which
deliberately carries the original model's state forward, so it is handled
separately here.

Accepted learner specifications
-------------------------------
1.  An unfitted (or fitted) scikit-learn compatible estimator. It is cloned
    before each fit, so the object passed in is never mutated. This covers
    RandomForest, GradientBoosting, LogisticRegression, XGBoost and LightGBM
    via their sklearn wrappers.

2.  A zero-argument callable returning a fresh model. Required for Keras and
    PyTorch, where architecture and compilation cannot be recovered by
    cloning. The returned object is dispatched on its type.

Prediction semantics
--------------------
predict_fn must return the conditional mean E[Y | x]; for binary
classification that is P(Y = 1 | x), never a class label. Class labels are
discrete and their finite differences are identically zero.
"""

import copy
import numpy as np
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Cloning
# ---------------------------------------------------------------------------

def clone_estimator(estimator):
    """
    Return an unfitted copy of a scikit-learn compatible estimator.

    Uses sklearn.base.clone when scikit-learn is installed, and reconstructs
    from get_params otherwise so that the core package keeps working without
    it.

    Parameters
    ----------
    estimator : object
        Estimator exposing get_params, or any deep-copyable object.

    Returns
    -------
    object
        A fresh, unfitted estimator.
    """
    if hasattr(estimator, 'get_params'):
        try:
            from sklearn.base import clone as sk_clone
            return sk_clone(estimator)
        except Exception:
            params = estimator.get_params(deep=False)
            return type(estimator)(**copy.deepcopy(params))
    return copy.deepcopy(estimator)


# ---------------------------------------------------------------------------
# Prediction wrappers
# ---------------------------------------------------------------------------

def _squeeze(pred: np.ndarray) -> np.ndarray:
    """Reduce a model output to a 1-D array of conditional means."""
    pred = np.asarray(pred, dtype=float)
    if pred.ndim == 2 and pred.shape[1] == 1:
        return pred[:, 0]
    if pred.ndim == 2 and pred.shape[1] == 2:
        return pred[:, 1]
    if pred.ndim > 1:
        return pred.reshape(pred.shape[0], -1)[:, 0]
    return pred


def make_predict_fn(fitted) -> Callable:
    """
    Build predict_fn(X) -> (n_obs,) for a fitted model of any supported type.

    Parameters
    ----------
    fitted : object
        A fitted model.

    Returns
    -------
    Callable
        predict_fn(X) returning conditional means.
    """
    # sklearn-style classifier
    if hasattr(fitted, 'predict_proba'):
        return lambda X: _squeeze(fitted.predict_proba(X))

    # Keras
    if _is_keras(fitted):
        return lambda X: _squeeze(
            fitted.predict(np.asarray(X, dtype=float), verbose=0)
        )

    # PyTorch
    if _is_torch(fitted):
        import torch

        def predict_fn(X):
            fitted.eval()
            with torch.no_grad():
                out = fitted(torch.tensor(np.asarray(X, dtype=float),
                                          dtype=torch.float32))
            return _squeeze(out.numpy())

        return predict_fn

    # sklearn-style regressor / anything else with predict
    if hasattr(fitted, 'predict'):
        return lambda X: _squeeze(fitted.predict(X))

    raise TypeError(
        f"Cannot build a prediction function for {type(fitted).__name__}."
    )


def _is_keras(obj) -> bool:
    try:
        import tensorflow as tf
        return isinstance(obj, tf.keras.Model)
    except Exception:
        return False


def _is_torch(obj) -> bool:
    try:
        import torch
        return isinstance(obj, torch.nn.Module)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------

class Learner:
    """
    Fits a learner from scratch on arbitrary subsets of the data.

    Parameters
    ----------
    spec : estimator or callable
        Either a scikit-learn compatible estimator (cloned before each fit)
        or a zero-argument callable returning a fresh model.
    n_epochs : int
        Training epochs, for Keras and PyTorch. Default 10.
    batch_size : int
        Batch size, for Keras and PyTorch. Default 32.
    optimizer_fn : Callable, optional
        PyTorch only. Takes model parameters, returns an optimizer.
        Default Adam at lr 1e-3.
    loss_fn : optional
        PyTorch only. Default BCELoss.
    """

    def __init__(
        self,
        spec,
        n_epochs: int = 10,
        batch_size: int = 32,
        optimizer_fn: Optional[Callable] = None,
        loss_fn=None,
    ):
        if not callable(spec) and not hasattr(spec, 'fit'):
            raise TypeError(
                "learner must be a scikit-learn compatible estimator or a "
                f"zero-argument callable returning one; got "
                f"{type(spec).__name__}."
            )

        # Keras models and torch modules carry trained weights that cloning
        # cannot strip, so accepting an instance would warm-start each fold
        # from a model that has already seen the held-out data. Demand a
        # factory instead.
        if _is_keras(spec) or _is_torch(spec):
            raise TypeError(
                f"Pass a zero-argument callable returning a fresh "
                f"{type(spec).__name__}, not a model instance. Cross-fitting "
                f"must train each fold from scratch; reusing a trained model "
                f"would leak the held-out fold into its own score."
            )

        self.spec = spec
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer_fn = optimizer_fn
        self.loss_fn = loss_fn

    def _fresh(self):
        """Produce an unfitted model instance."""
        # An estimator is callable in rare cases; prefer the estimator path.
        if hasattr(self.spec, 'fit'):
            return clone_estimator(self.spec)
        return self.spec()

    def fit(self, X: np.ndarray, y: np.ndarray) -> Callable:
        """
        Fit a fresh model on (X, y) and return its prediction function.

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_train, n_features).
        y : np.ndarray
            Training targets, shape (n_train,).

        Returns
        -------
        Callable
            predict_fn(X) -> np.ndarray of conditional means.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        model = self._fresh()

        if _is_keras(model):
            model.fit(
                X, y,
                epochs=self.n_epochs,
                batch_size=self.batch_size,
                verbose=0,
            )
        elif _is_torch(model):
            self._fit_torch(model, X, y)
        else:
            model.fit(X, y)

        return make_predict_fn(model)

    def _fit_torch(self, model, X: np.ndarray, y: np.ndarray) -> None:
        """Train a PyTorch module in place."""
        import torch

        loss_fn = self.loss_fn if self.loss_fn is not None else torch.nn.BCELoss()
        optimizer_fn = self.optimizer_fn
        if optimizer_fn is None:
            def optimizer_fn(params):
                return torch.optim.Adam(params, lr=1e-3)

        model.train()
        optimizer = optimizer_fn(model.parameters())

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        n = X_t.shape[0]
        n_batches = max(1, n // self.batch_size)

        for _ in range(self.n_epochs):
            perm = torch.randperm(n)
            X_s, y_s = X_t[perm], y_t[perm]
            for b in range(n_batches):
                start = b * self.batch_size
                end = start + self.batch_size
                optimizer.zero_grad()
                out = model(X_s[start:end])
                if out.ndim == 2 and out.shape[1] == 1:
                    out = out.squeeze(dim=1)
                elif out.ndim == 2 and out.shape[1] == 2:
                    out = out[:, 1]
                loss = loss_fn(out, y_s[start:end])
                loss.backward()
                optimizer.step()

        model.eval()
