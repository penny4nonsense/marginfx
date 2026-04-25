"""
engines/tensorflow.py
---------------------
TensorFlow engine for marginfx.

Key difference from the sklearn engine:
    - Uses tf.GradientTape for exact gradients instead of finite differences
    - This is faster and more accurate than numerical approximation
    - Warm-start via continued training on bootstrap sample (default 10 epochs)

Gradient computation:
    - Regression: dy/dx directly from GradientTape
    - Classification: dP(y=1)/dx from GradientTape — exactly what we want
      for marginal effects on probability, no predict_proba needed

Warm-start:
    - Keep original model weights as initialization
    - Continue training on bootstrap sample for n_epochs (default 10)
    - User-configurable via make_fit_fn(model, n_epochs=10)
"""

import numpy as np
import copy
from typing import Callable, Tuple, Optional


# ---------------------------------------------------------------------------
# Gradient-based predict function
# ---------------------------------------------------------------------------

def make_predict_fn(model) -> Callable:
    """
    Build a predict_fn for a fitted TensorFlow/Keras model.

    Returns standard predictions (not gradients). The gradient computation
    lives in the AME engine override — see make_gradient_ame_fn() below.

    For binary classification (sigmoid output): returns P(y=1)
    For regression: returns predicted values directly

    Parameters
    ----------
    model : tf.keras.Model
        A fitted Keras model.

    Returns
    -------
    Callable
        predict_fn(X) -> np.ndarray of shape (n_obs,)
    """
    import tensorflow as tf

    def predict_fn(X):
        X_tensor = tf.cast(tf.constant(X), dtype=tf.float32)
        predictions = model(X_tensor, training=False).numpy()

        # Squeeze to 1D if output shape is (n, 1)
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            return predictions.squeeze(axis=1)

        # Binary classification with 2 output units + softmax
        if predictions.ndim == 2 and predictions.shape[1] == 2:
            return predictions[:, 1]

        return predictions

    return predict_fn


# ---------------------------------------------------------------------------
# Exact gradient computation via GradientTape
# ---------------------------------------------------------------------------

def make_gradient_ame_fn(model) -> Callable:
    """
    Build an AME function using exact gradients from tf.GradientTape.

    This replaces the finite difference computation in core.py for
    TensorFlow models. Exact gradients are faster and more accurate.

    For categorical features, falls back to first differences since
    gradients with respect to discrete inputs are not meaningful.

    Parameters
    ----------
    model : tf.keras.Model
        A fitted Keras model.

    Returns
    -------
    Callable
        gradient_ame_fn(X, feature_idx, is_categorical, h) -> np.ndarray
        Returns pointwise marginal effects for a single feature, shape (n_obs,)
    """
    import tensorflow as tf

    def gradient_ame_fn(
        X: np.ndarray,
        feature_idx: int,
        is_categorical: bool = False,
        h: float = 1e-4,
    ) -> np.ndarray:

        # Categorical features: first difference, no gradient needed
        if is_categorical:
            X_0 = X.copy()
            X_1 = X.copy()
            X_0[:, feature_idx] = 0.0
            X_1[:, feature_idx] = 1.0

            X_0_tensor = tf.cast(tf.constant(X_0), dtype=tf.float32)
            X_1_tensor = tf.cast(tf.constant(X_1), dtype=tf.float32)

            pred_0 = model(X_0_tensor, training=False).numpy()
            pred_1 = model(X_1_tensor, training=False).numpy()

            # Squeeze outputs
            pred_0 = _squeeze_output(pred_0)
            pred_1 = _squeeze_output(pred_1)

            return pred_1 - pred_0

        # Continuous features: exact gradient via GradientTape
        X_tensor = tf.cast(tf.Variable(X), dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = model(X_tensor, training=False)
            predictions = _squeeze_output_tensor(predictions)

        # Gradient of output with respect to all inputs
        # Shape: (n_obs, n_features)
        grads = tape.gradient(predictions, X_tensor).numpy()

        # Return gradient for the requested feature only
        # Shape: (n_obs,)
        return grads[:, feature_idx]

    return gradient_ame_fn


def _squeeze_output(predictions: np.ndarray) -> np.ndarray:
    """Squeeze model output to 1D array."""
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        return predictions.squeeze(axis=1)
    if predictions.ndim == 2 and predictions.shape[1] == 2:
        return predictions[:, 1]
    return predictions


def _squeeze_output_tensor(predictions):
    """Squeeze tensor output for GradientTape compatibility."""
    import tensorflow as tf
    if len(predictions.shape) == 2 and predictions.shape[1] == 1:
        return tf.squeeze(predictions, axis=1)
    if len(predictions.shape) == 2 and predictions.shape[1] == 2:
        return predictions[:, 1]
    return predictions


# ---------------------------------------------------------------------------
# Warm-start fit function
# ---------------------------------------------------------------------------

def make_fit_fn(
    model,
    n_epochs: int = 10,
    batch_size: int = 32,
    verbose: int = 0,
) -> Callable:
    """
    Build a fit_fn for a TensorFlow/Keras model.

    Warm-starts by continuing training from the original model weights
    on the bootstrap sample. Runs for n_epochs (default 10) — enough
    to adapt to the new sample without forgetting the original fit.

    The model must already be compiled (optimizer, loss, metrics).
    The bootstrap refit inherits the same compilation.

    Parameters
    ----------
    model : tf.keras.Model
        Original fitted and compiled Keras model.
    n_epochs : int
        Number of epochs to train on each bootstrap sample. Default 10.
    batch_size : int
        Batch size for bootstrap refit. Default 32.
    verbose : int
        Keras verbosity. 0 = silent. Default 0.

    Returns
    -------
    Callable
        fit_fn(model, X_boot, y_boot) -> fitted_model
    """
    import tensorflow as tf

    def fit_fn(current_model, X_boot: np.ndarray, y_boot: np.ndarray):
        # Clone model architecture and copy weights from current_model
        # This preserves the warm-start without mutating the original
        new_model = tf.keras.models.clone_model(current_model)
        new_model.set_weights(current_model.get_weights())

        # Inherit compilation from original model
        new_model.compile(
            optimizer=current_model.optimizer.__class__(
                learning_rate=current_model.optimizer.learning_rate
            ),
            loss=current_model.loss,
            metrics=current_model.compiled_metrics._metrics,
        )

        X_tensor = tf.cast(tf.constant(X_boot), dtype=tf.float32)
        y_tensor = tf.cast(tf.constant(y_boot), dtype=tf.float32)

        new_model.fit(
            X_tensor,
            y_tensor,
            epochs=n_epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

        return new_model

    return fit_fn


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_engine(
    model,
    n_epochs: int = 10,
    batch_size: int = 32,
) -> Tuple[Callable, Callable, Optional[Callable]]:
    """
    Get predict_fn, fit_fn, and gradient_ame_fn for a TensorFlow/Keras model.

    The gradient_ame_fn replaces finite differences in core.py with exact
    gradients from tf.GradientTape — faster and more accurate.

    Parameters
    ----------
    model : tf.keras.Model
        Fitted and compiled Keras model.
    n_epochs : int
        Bootstrap refit epochs. Default 10.
    batch_size : int
        Bootstrap refit batch size. Default 32.

    Returns
    -------
    Tuple[Callable, Callable, Callable]
        (predict_fn, fit_fn, gradient_ame_fn)

        predict_fn(X) -> np.ndarray
            Standard predictions for output and visualization.

        fit_fn(model, X_boot, y_boot) -> fitted_model
            Warm-start refit on bootstrap sample.

        gradient_ame_fn(X, feature_idx, is_categorical, h) -> np.ndarray
            Exact pointwise marginal effects via GradientTape.
            Pass this to all_ames() to override finite differences.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from engines.tensorflow import get_engine
    >>> from bootstrap import bootstrap_ames
    >>>
    >>> model = tf.keras.models.load_model('my_model.keras')
    >>> predict_fn, fit_fn, gradient_ame_fn = get_engine(model)
    >>> result = bootstrap_ames(
    ...     model, X, y, fit_fn, predict_fn,
    ...     gradient_ame_fn=gradient_ame_fn
    ... )
    >>> result.summary()
    """
    predict_fn = make_predict_fn(model)
    fit_fn = make_fit_fn(model, n_epochs=n_epochs, batch_size=batch_size)
    gradient_ame_fn = make_gradient_ame_fn(model)

    return predict_fn, fit_fn, gradient_ame_fn
