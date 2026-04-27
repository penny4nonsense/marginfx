"""
tests/test_engines/test_sklearn.py
-----------------------------------
Unit tests for engines/sklearn.py.

Tests cover:
    - predict_fn autodetection (classifier vs regressor)
    - predict_fn output shape
    - fit_fn warm-start behavior
    - fit_fn cold refit fallback
    - get_engine() returns correct callables
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from marginfx.engines.sklearn import get_engine, make_predict_fn, make_fit_fn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def classification_data(rng):
    """Binary classification dataset."""
    X = rng.standard_normal((200, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    return X, y


@pytest.fixture
def regression_data(rng):
    """Regression dataset."""
    X = rng.standard_normal((200, 4))
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + rng.standard_normal(200) * 0.5
    return X, y


@pytest.fixture
def fitted_rf_classifier(classification_data):
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def fitted_rf_regressor(regression_data):
    X, y = regression_data
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def fitted_logistic(classification_data):
    X, y = classification_data
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def fitted_linear(regression_data):
    X, y = regression_data
    model = LinearRegression()
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def fitted_svc(classification_data):
    """SVC has no predict_proba by default and no warm_start."""
    X, y = classification_data
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# predict_fn tests
# ---------------------------------------------------------------------------

class TestPredictFn:

    def test_classifier_uses_predict_proba(self, fitted_rf_classifier):
        """RandomForestClassifier should use predict_proba[:, 1]."""
        model, X, y = fitted_rf_classifier
        predict_fn = make_predict_fn(model)
        result = predict_fn(X)

        # Should be probabilities in [0, 1]
        assert result.shape == (X.shape[0],)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_regressor_uses_predict(self, fitted_rf_regressor):
        """RandomForestRegressor should use predict() directly."""
        model, X, y = regression_data = fitted_rf_regressor
        predict_fn = make_predict_fn(model)
        result = predict_fn(X)

        assert result.shape == (X.shape[0],)
        # Regression output can be any real value
        assert not (result.min() >= 0 and result.max() <= 1)

    def test_logistic_uses_predict_proba(self, fitted_logistic):
        """LogisticRegression should use predict_proba[:, 1]."""
        model, X, y = fitted_logistic
        predict_fn = make_predict_fn(model)
        result = predict_fn(X)

        assert result.shape == (X.shape[0],)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_linear_regression_uses_predict(self, fitted_linear):
        """LinearRegression should use predict() directly."""
        model, X, y = fitted_linear
        predict_fn = make_predict_fn(model)
        result = predict_fn(X)

        assert result.shape == (X.shape[0],)

    def test_output_is_1d(self, fitted_rf_classifier):
        """predict_fn output should always be 1D."""
        model, X, y = fitted_rf_classifier
        predict_fn = make_predict_fn(model)
        result = predict_fn(X)
        assert result.ndim == 1

    def test_classifier_probabilities_sum(self, fitted_logistic):
        """Returned probabilities should be in (0, 1) for logistic."""
        model, X, y = fitted_logistic
        predict_fn = make_predict_fn(model)
        result = predict_fn(X)
        assert np.all(result > 0) and np.all(result < 1)


# ---------------------------------------------------------------------------
# fit_fn tests
# ---------------------------------------------------------------------------

class TestFitFn:

    def test_warm_start_fires_for_random_forest(
        self, fitted_rf_classifier, classification_data
    ):
        """RandomForest has warm_start — fit_fn should use it."""
        model, X, y = fitted_rf_classifier
        fit_fn = make_fit_fn(model)

        # Resample
        idx = np.random.default_rng(42).integers(0, X.shape[0], size=X.shape[0])
        X_boot, y_boot = X[idx], y[idx]

        new_model = fit_fn(model, X_boot, y_boot)

        # New model should be able to predict
        assert hasattr(new_model, 'predict')
        assert new_model.predict(X).shape == (X.shape[0],)

    def test_cold_refit_for_svc(self, fitted_svc, classification_data):
        """SVC has no warm_start — fit_fn should refit cold silently."""
        model, X, y = fitted_svc
        fit_fn = make_fit_fn(model)

        idx = np.random.default_rng(42).integers(0, X.shape[0], size=X.shape[0])
        X_boot, y_boot = X[idx], y[idx]

        # Should not raise, should not warn
        new_model = fit_fn(model, X_boot, y_boot)
        assert hasattr(new_model, 'predict')

    def test_fit_fn_does_not_mutate_original(
        self, fitted_rf_classifier, classification_data
    ):
        """fit_fn should not mutate the original model."""
        model, X, y = fitted_rf_classifier
        original_predictions = model.predict_proba(X)[:, 1].copy()

        fit_fn = make_fit_fn(model)
        idx = np.random.default_rng(42).integers(0, X.shape[0], size=X.shape[0])
        X_boot, y_boot = X[idx], y[idx]
        _ = fit_fn(model, X_boot, y_boot)

        # Original model predictions should be unchanged
        new_predictions = model.predict_proba(X)[:, 1]
        assert np.allclose(original_predictions, new_predictions)

    def test_fitted_model_can_predict(
        self, fitted_logistic, classification_data
    ):
        """Model returned by fit_fn should be able to predict."""
        model, X, y = fitted_logistic
        fit_fn = make_fit_fn(model)

        idx = np.random.default_rng(42).integers(0, X.shape[0], size=X.shape[0])
        X_boot, y_boot = X[idx], y[idx]

        new_model = fit_fn(model, X_boot, y_boot)
        result = new_model.predict_proba(X)[:, 1]

        assert result.shape == (X.shape[0],)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ---------------------------------------------------------------------------
# get_engine tests
# ---------------------------------------------------------------------------

class TestGetEngine:

    def test_returns_two_callables(self, fitted_rf_classifier):
        """get_engine should return (predict_fn, fit_fn)."""
        model, X, y = fitted_rf_classifier
        result = get_engine(model)

        assert len(result) == 2
        predict_fn, fit_fn = result
        assert callable(predict_fn)
        assert callable(fit_fn)

    def test_predict_fn_works(self, fitted_rf_classifier):
        """predict_fn from get_engine should return correct shape."""
        model, X, y = fitted_rf_classifier
        predict_fn, fit_fn = get_engine(model)
        result = predict_fn(X)
        assert result.shape == (X.shape[0],)

    def test_fit_fn_works(self, fitted_rf_classifier):
        """fit_fn from get_engine should return a fitted model."""
        model, X, y = fitted_rf_classifier
        predict_fn, fit_fn = get_engine(model)

        idx = np.random.default_rng(42).integers(0, X.shape[0], size=X.shape[0])
        X_boot, y_boot = X[idx], y[idx]

        new_model = fit_fn(model, X_boot, y_boot)
        assert hasattr(new_model, 'predict')

    def test_works_with_xgboost(self, classification_data):
        """get_engine should work with XGBoost classifier."""
        pytest.importorskip('xgboost')
        import xgboost as xgb

        X, y = classification_data
        model = xgb.XGBClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        predict_fn, fit_fn = get_engine(model)
        result = predict_fn(X)

        assert result.shape == (X.shape[0],)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_works_with_lightgbm(self, classification_data):
        """get_engine should work with LightGBM classifier."""
        pytest.importorskip('lightgbm')
        import lightgbm as lgb

        X, y = classification_data
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        predict_fn, fit_fn = get_engine(model)
        result = predict_fn(X)

        assert result.shape == (X.shape[0],)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
