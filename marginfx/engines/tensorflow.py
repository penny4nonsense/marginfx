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
    - Rebuild model from config each bootstrap replicate
    - Clear TF session before each rebuild to prevent memory accumulation
    - Set weights from original model for warm-start
"""

import numpy as np
from typing import Callable, Tuple, Optional


# ---------------------------------------------------------------------------
# Gradient-based predict function
# ---------------------------------------------------------------------------

def make_predict_fn(model) -> Callable:
    """
    Build a predict_fn for a fitted TensorFlow/Keras model.

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

        if predictions.ndim == 2 and predictions.shape[1] == 1:
            return predictions.squeeze(axis=1)
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

    Parameters
    ----------
    model : tf.keras.Model
        A fitted Keras model.

    Returns
    -------
    Callable
        gradient_ame_fn(X, feature_idx, is_categorical, h) -> np.ndarray
    """
    import tensorflow as tf

    def gradient_ame_fn(
        X: np.ndarray,
        feature_idx: int,
        is_categorical: bool = False,
        h: float = 1e-4,
    ) -> np.ndarray:

        if is_categorical:
            X_0 = X.copy()
            X_1 = X.copy()
            X_0[:, feature_idx] = 0.0
            X_1[:, feature_idx] = 1.0

            X_0_tensor = tf.cast(tf.constant(X_0), dtype=tf.float32)
            X_1_tensor = tf.cast(tf.constant(X_1), dtype=tf.float32)

            pred_0 = _squeeze_output(model(X_0_tensor, training=False).numpy())
            pred_1 = _squeeze_output(model(X_1_tensor, training=False).numpy())

            return pred_1 - pred_0

        X_tensor = tf.cast(tf.Variable(X), dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = _squeeze_output_tensor(model(X_tensor, training=False))

        grads = tape.gradient(predictions, X_tensor).numpy()
        return grads[:, feature_idx]

    return gradient_ame_fn


def _squeeze_output(predictions: np.ndarray) -> np.ndarray:
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        return predictions.squeeze(axis=1)
    if predictions.ndim == 2 and predictions.shape[1] == 2:
        return predictions[:, 1]
    return predictions


def _squeeze_output_tensor(predictions):
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

    Rebuilds the model from config each bootstrap replicate and clears
    the TF session beforehand to prevent memory accumulation.

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
    # Capture config and settings from original model at creation time
    # so they are available in the closure even after clear_session()
    original_config = model.get_config()
    original_weights = model.get_weights()
    original_lr = float(model.optimizer.learning_rate.numpy())
    original_loss = model.loss

    def fit_fn(current_model, X_boot: np.ndarray, y_boot: np.ndarray):
        import gc
        import tensorflow as tf

        # Get current weights for warm-start
        try:
            weights = current_model.get_weights()
            lr = float(current_model.optimizer.learning_rate.numpy())
            config = current_model.get_config()
            loss = current_model.loss
        except Exception:
            # Fall back to original model settings if current_model is unavailable
            weights = original_weights
            lr = original_lr
            config = original_config
            loss = original_loss

        # Clear session to free accumulated graph memory
        tf.keras.backend.clear_session()
        tf.keras.utils.disable_interactive_logging()

        # Rebuild from config with warm-start weights
        new_model = tf.keras.Sequential.from_config(config)
        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=loss,
        )
        new_model.set_weights(weights)

        X_tensor = tf.cast(tf.constant(X_boot), dtype=tf.float32)
        y_tensor = tf.cast(tf.constant(y_boot), dtype=tf.float32)

        new_model.fit(
            X_tensor,
            y_tensor,
            epochs=n_epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

        gc.collect()
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
    """
    predict_fn = make_predict_fn(model)
    fit_fn = make_fit_fn(model, n_epochs=n_epochs, batch_size=batch_size, verbose=0)
    gradient_ame_fn = make_gradient_ame_fn(model)

    return predict_fn, fit_fn, gradient_ame_fn