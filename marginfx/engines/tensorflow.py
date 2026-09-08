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

        # Architecture and training settings are reused; the fitted weights
        # are not. Restarting from the full-sample weights would carry that
        # fit into every replicate and understate how far the fitted function
        # moves with the data, which is the whole quantity being measured.
        try:
            lr = float(current_model.optimizer.learning_rate.numpy())
            config = current_model.get_config()
            loss = current_model.loss
        except Exception:
            # Fall back to original model settings if current_model is unavailable
            lr = original_lr
            config = original_config
            loss = original_loss

        # Clear session to free accumulated graph memory
        tf.keras.backend.clear_session()
        tf.keras.utils.disable_interactive_logging()

        # Rebuild from config with freshly initialized weights
        new_model = tf.keras.Sequential.from_config(config)
        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=loss,
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
) -> Tuple[Callable, Callable]:
    """
    Get predict_fn and fit_fn for a TensorFlow/Keras model.

    Used by the bootstrap diagnostic path, which warm-starts from an already
    fitted model. Cross-fitting uses learner.Learner instead, which trains
    each fold from scratch.

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
    Tuple[Callable, Callable]
        (predict_fn, fit_fn)
    """
    predict_fn = make_predict_fn(model)
    fit_fn = make_fit_fn(model, n_epochs=n_epochs, batch_size=batch_size, verbose=0)

    return predict_fn, fit_fn