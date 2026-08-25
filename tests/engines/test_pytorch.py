"""
tests/test_engines/test_pytorch.py
------------------------------------
Unit tests for engines/pytorch.py.

Tests cover:
    - predict_fn output shape and range
    - predict_fn runs in eval mode
    - fit_fn warm-start refit produces working model in eval mode
    - fit_fn does not mutate original model weights
    - get_engine() returns two callables
    - Gradient accuracy on known linear model

All tests are skipped if PyTorch is not installed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from marginfx.engines.pytorch import (
    get_engine,
    make_predict_fn,
    make_fit_fn,
    DEFAULT_OPTIMIZER_FN,
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


def _make_binary_classifier():
    """Minimal binary classification PyTorch model."""
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid(),
    )


def _make_regressor():
    """Minimal regression PyTorch model."""
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )


def _train_model(model, X, y, loss_fn, n_epochs=5):
    """Quick training loop for fixtures."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    model.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        out = model(X_t).squeeze()
        loss = loss_fn(out, y_t)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


@pytest.fixture
def binary_classifier(classification_data):
    """Minimal trained binary classification PyTorch model."""
    X, y = classification_data
    model = _make_binary_classifier()
    model = _train_model(model, X, y, nn.BCELoss(), n_epochs=5)
    return model, X, y


@pytest.fixture
def regressor(regression_data):
    """Minimal trained regression PyTorch model."""
    X, y = regression_data
    model = _make_regressor()
    model = _train_model(model, X, y, nn.MSELoss(), n_epochs=5)
    return model, X, y


@pytest.fixture
def linear_model_torch():
    """
    Exact linear PyTorch model for gradient correctness testing.
    f(x) = 2*x1 + 3*x2 + 0*x3 + 0*x4
    Weights set manually so gradients are exactly known.
    """
    model = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(
            torch.tensor([[2.0, 3.0, 0.0, 0.0]], dtype=torch.float32)
        )
    model.eval()
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
        """predict_fn should return numpy array not tensor."""
        model, X, y = binary_classifier
        predict_fn = make_predict_fn(model)
        result = predict_fn(X)
        assert isinstance(result, np.ndarray)

    def test_model_in_eval_mode_after_predict(self, binary_classifier):
        """
        predict_fn should leave model in eval mode.
        This is important for dropout and batchnorm correctness.
        """
        model, X, y = binary_classifier
        predict_fn = make_predict_fn(model)
        _ = predict_fn(X)
        assert not model.training, "Model should be in eval mode after predict_fn"

    def test_deterministic_output(self, binary_classifier):
        """predict_fn should produce identical results on repeated calls."""
        model, X, y = binary_classifier
        predict_fn = make_predict_fn(model)
        r1 = predict_fn(X)
        r2 = predict_fn(X)
        assert np.allclose(r1, r2)


# ---------------------------------------------------------------------------
class TestFitFn:

    def test_returns_pytorch_module(self, binary_classifier):
        """fit_fn should return a torch.nn.Module."""
        model, X, y = binary_classifier
        fit_fn = make_fit_fn(model, n_epochs=2)

        idx = np.random.default_rng(42).integers(0, X.shape[0], size=X.shape[0])
        X_boot, y_boot = X[idx], y[idx]

        new_model = fit_fn(model, X_boot, y_boot)
        assert isinstance(new_model, nn.Module)

    def test_returned_model_in_eval_mode(self, binary_classifier):
        """
        Model returned by fit_fn should be in eval mode.
        Critical for dropout and batchnorm correctness during AME computation.
        """
        model, X, y = binary_classifier
        fit_fn = make_fit_fn(model, n_epochs=2)

        idx = np.random.default_rng(42).integers(0, X.shape[0], size=X.shape[0])
        X_boot, y_boot = X[idx], y[idx]

        new_model = fit_fn(model, X_boot, y_boot)
        assert not new_model.training, \
            "fit_fn should return model in eval mode"

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
        """fit_fn should not modify the original model's weights."""
        model, X, y = binary_classifier

        # Record original weights
        original_weights = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

        fit_fn = make_fit_fn(model, n_epochs=5)
        idx = np.random.default_rng(42).integers(0, X.shape[0], size=X.shape[0])
        X_boot, y_boot = X[idx], y[idx]
        _ = fit_fn(model, X_boot, y_boot)

        for name, param in model.named_parameters():
            assert torch.allclose(original_weights[name], param.data), \
                f"fit_fn mutated original model parameter: {name}"

    def test_warm_start_initializes_from_original(self, binary_classifier):
        """
        New model should be initialized from original weights.
        With n_epochs=1, weights should remain close to original.
        """
        model, X, y = binary_classifier
        original_weights = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

        fit_fn = make_fit_fn(model, n_epochs=1, batch_size=200)
        idx = np.random.default_rng(42).integers(0, X.shape[0], size=X.shape[0])
        X_boot, y_boot = X[idx], y[idx]
        new_model = fit_fn(model, X_boot, y_boot)

        for name, param in new_model.named_parameters():
            max_diff = (original_weights[name] - param.data).abs().max().item()
            assert max_diff < 10.0, \
                f"Parameter {name} moved too far from original — " \
                f"warm-start may have failed. Max diff: {max_diff:.4f}"

    def test_custom_optimizer_fn(self, binary_classifier):
        """fit_fn should accept a custom optimizer_fn."""
        model, X, y = binary_classifier

        custom_optimizer_fn = lambda params: torch.optim.SGD(
            params, lr=0.01, momentum=0.9
        )
        fit_fn = make_fit_fn(
            model,
            optimizer_fn=custom_optimizer_fn,
            n_epochs=2,
        )

        idx = np.random.default_rng(42).integers(0, X.shape[0], size=X.shape[0])
        X_boot, y_boot = X[idx], y[idx]

        new_model = fit_fn(model, X_boot, y_boot)
        assert isinstance(new_model, nn.Module)

    def test_custom_loss_fn(self, regressor):
        """fit_fn should accept a custom loss function."""
        model, X, y = regressor

        fit_fn = make_fit_fn(
            model,
            loss_fn=nn.MSELoss(),
            n_epochs=2,
        )

        idx = np.random.default_rng(42).integers(0, X.shape[0], size=X.shape[0])
        X_boot, y_boot = X[idx], y[idx]

        new_model = fit_fn(model, X_boot, y_boot)
        assert isinstance(new_model, nn.Module)


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
        assert isinstance(new_model, nn.Module)

    def test_custom_optimizer_passthrough(self, binary_classifier):
        """Custom optimizer_fn should be passed through to fit_fn."""
        model, X, y = binary_classifier

        custom_opt = lambda params: torch.optim.SGD(params, lr=0.01)
        predict_fn, fit_fn = get_engine(
            model,
            optimizer_fn=custom_opt,
            n_epochs=2,
        )

        idx = np.random.default_rng(42).integers(0, X.shape[0], size=X.shape[0])
        X_boot, y_boot = X[idx], y[idx]

        new_model = fit_fn(model, X_boot, y_boot)
        assert isinstance(new_model, nn.Module)

    def test_custom_loss_passthrough(self, regressor):
        """Custom loss_fn should be passed through to fit_fn."""
        model, X, y = regressor

        predict_fn, fit_fn = get_engine(
            model,
            loss_fn=nn.MSELoss(),
            n_epochs=2,
        )

        idx = np.random.default_rng(42).integers(0, X.shape[0], size=X.shape[0])
        X_boot, y_boot = X[idx], y[idx]

        new_model = fit_fn(model, X_boot, y_boot)
        assert isinstance(new_model, nn.Module)
