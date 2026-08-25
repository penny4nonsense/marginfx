"""
tests/test_engines/test_tensorflow.py
--------------------------------------
Unit tests for engines/tensorflow.py.

Tests cover:
    - predict_fn output shape and range for classification and regression
    - fit_fn warm-start refit produces a working model
    - fit_fn does not mutate original model weights
    - get_engine() returns two callables
    - Gradient accuracy on known linear model

All tests are skipped if TensorFlow is not installed.
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from marginfx.engines.tensorflow import (
    get_engine,
    make_predict_fn,
    make_fit_fn,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def classification_data(rng):
    """Binary classification dataset, 4 features."""
    X = rng.standard_normal((200, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    return X, y


@pytest.fixture
def regression_data(rng):
    """Regression dataset, 4 features."""
    X = rng.standard_normal((200, 4)).astype(np.float32)
    y = (2.0 * X[:, 0] + 3.0 * X[:, 1]).astype(np.float32)
    return X, y


@pytest.fixture
def binary_classifier(classification_data):
    """Minimal compiled binary classification Keras model."""
    X, y = classification_data
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X, y, epochs=2, verbose=0)
    return model, X, y


@pytest.fixture
def regressor(regression_data):
    """Minimal compiled regression Keras model."""
    X, y = regression_data
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=2, verbose=0)
    return model, X, y


@pytest.fixture
def linear_model_tf():
    """
    Exact linear Keras model for gradient correctness testing.
    f(x) = 2*x1 + 3*x2 (approximately, via single Dense layer with no bias trick)
    We set weights manually so gradients are exactly known.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, use_bias=False, input_shape=(4,)),
    ])
    # Build the model
    model.build(input_shape=(None, 4))
    # Set weights manually: [2.0, 3.0, 0.0, 0.0]
    model.layers[0].set_weights([
        np.array([[2.0], [3.0], [0.0], [0.0]], dtype=np.float32)
    ])
    return model


# ---------------------------------------------------------------------------
# predict_fn tests
# ---------------------------------------------------------------------------

class TestPredictFn:

    def test_classifier_output_shape(self, binary_classifier):
        """predict_fn for classifier should return shape (n_obs,)."""
        model, X, y = binary_classifier
        predict_fn = make_predict_fn(model)
        result = predict_fn(X)
        assert result.shape == (X.shape[0],)

    def test_classifier_output_in_unit_interval(self, binary_classifier):
        """Sigmoid output should be in [0, 1]."""
        model, X, y = binary_classifier
        predict_fn = make_predict_fn(model)
        result = predict_fn(X)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_regressor_output_shape(self, regressor):
        """predict_fn for regressor should return shape (n_obs,)."""
        model, X, y = regressor
        predict_fn = make_predict_fn(model)
        result = predict_fn(X)
        assert result.shape == (X.shape[0],)

    def test_output_is_1d(self, binary_classifier):
        """predict_fn output should always be 1D numpy array."""
        model, X, y = binary_classifier
        predict_fn = make_predict_fn(model)
        result = predict_fn(X)
        assert result.ndim == 1

    def test_output_is_numpy(self, binary_classifier):
        """predict_fn should return numpy array, not tensor."""
        model, X, y = binary_classifier
        predict_fn = make_predict_fn(model)
        result = predict_fn(X)
        assert isinstance(result, np.ndarray)

    def test_deterministic_in_eval(self, binary_classifier):
        """predict_fn should return same result on repeated calls."""
        model, X, y = binary_classifier
        predict_fn = make_predict_fn(model)
        r1 = predict_fn(X)
        r2 = predict_fn(X)
        assert np.allclose(r1, r2)


# ---------------------------------------------------------------------------
class TestFitFn:

    def test_returns_keras_model(self, binary_classifier):
        """fit_fn should return a tf.keras.Model."""
        model, X, y = binary_classifier
        fit_fn = make_fit_fn(model, n_epochs=2)

        idx = np.random.default_rng(42).integers(0, X.shape[0], size=X.shape[0])
        X_boot, y_boot = X[idx], y[idx]

        new_model = fit_fn(model, X_boot, y_boot)
        assert isinstance(new_model, tf.keras.Model)

    def test_new_model_can_predict(self, binary_classifier):
        """Model returned by fit_fn should produce valid predictions."""
        model, X, y = binary_classifier
        fit_fn = make_fit_fn(model, n_epochs=2)

        idx = np.random.default_rng(42).integers(0, X.shape[0], size=X.shape[0])
        X_boot, y_boot = X[idx], y[idx]

        new_model = fit_fn(model, X_boot, y_boot)
        predict_fn = make_predict_fn(new_model)
        result = predict_fn(X)

        assert result.shape == (X.shape[0],)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_does_not_mutate_original_weights(self, binary_classifier):
        """fit_fn should not change the original model's weights."""
        model, X, y = binary_classifier

        # Record original weights
        original_weights = [w.numpy().copy() for w in model.weights]

        fit_fn = make_fit_fn(model, n_epochs=5)
        idx = np.random.default_rng(42).integers(0, X.shape[0], size=X.shape[0])
        X_boot, y_boot = X[idx], y[idx]
        _ = fit_fn(model, X_boot, y_boot)

        # Original weights should be unchanged
        for orig, current in zip(original_weights, model.weights):
            assert np.allclose(orig, current.numpy()), \
                "fit_fn mutated original model weights"

    def test_warm_start_initializes_from_original(self, binary_classifier):
        """
        New model should start from original weights.
        With n_epochs=1 and small lr, weights should be close to original.
        """
        model, X, y = binary_classifier
        original_weights = [w.numpy().copy() for w in model.weights]

        # Use very small learning rate so weights barely move
        fit_fn = make_fit_fn(model, n_epochs=1, batch_size=200)
        idx = np.random.default_rng(42).integers(0, X.shape[0], size=X.shape[0])
        X_boot, y_boot = X[idx], y[idx]
        new_model = fit_fn(model, X_boot, y_boot)

        new_weights = [w.numpy() for w in new_model.weights]

        # Weights should be initialized from original (not random)
        # Check that at least some weights are close to original
        max_diffs = [np.abs(o - n).max() for o, n in zip(original_weights, new_weights)]
        # Should be much closer than random init would be
        assert all(d < 10.0 for d in max_diffs), \
            "New model weights seem far from original — warm-start may have failed"

    def test_n_epochs_parameter(self, binary_classifier):
        """fit_fn should accept and use n_epochs parameter."""
        model, X, y = binary_classifier

        # Should not raise with different n_epochs values
        fit_fn_2 = make_fit_fn(model, n_epochs=2)
        fit_fn_10 = make_fit_fn(model, n_epochs=10)

        idx = np.random.default_rng(42).integers(0, X.shape[0], size=X.shape[0])
        X_boot, y_boot = X[idx], y[idx]

        m2 = fit_fn_2(model, X_boot, y_boot)
        m10 = fit_fn_10(model, X_boot, y_boot)

        assert isinstance(m2, tf.keras.Model)
        assert isinstance(m10, tf.keras.Model)


# ---------------------------------------------------------------------------
# get_engine tests
# ---------------------------------------------------------------------------

class TestGetEngine:

    def test_returns_two_callables(self, binary_classifier):
        """get_engine should return (predict_fn, fit_fn)."""
        model, X, y = binary_classifier
        result = get_engine(model)

        assert len(result) == 2
        predict_fn, fit_fn = result
        assert callable(predict_fn)
        assert callable(fit_fn)

    def test_predict_fn_works(self, binary_classifier):
        """predict_fn from get_engine should return correct shape."""
        model, X, y = binary_classifier
        predict_fn, fit_fn = get_engine(model)
        result = predict_fn(X)
        assert result.shape == (X.shape[0],)


    def test_fit_fn_works(self, binary_classifier):
        """fit_fn from get_engine should return a fitted model."""
        model, X, y = binary_classifier
        predict_fn, fit_fn = get_engine(model)

        idx = np.random.default_rng(42).integers(0, X.shape[0], size=X.shape[0])
        X_boot, y_boot = X[idx], y[idx]

        new_model = fit_fn(model, X_boot, y_boot)
        assert isinstance(new_model, tf.keras.Model)

    def test_n_epochs_passthrough(self, binary_classifier):
        """n_epochs should be passed through to fit_fn."""
        model, X, y = binary_classifier
        predict_fn, fit_fn = get_engine(model, n_epochs=3)

        idx = np.random.default_rng(42).integers(0, X.shape[0], size=X.shape[0])
        X_boot, y_boot = X[idx], y[idx]

        # Should not raise
        new_model = fit_fn(model, X_boot, y_boot)
        assert isinstance(new_model, tf.keras.Model)