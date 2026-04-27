"""
tests/test_bootstrap.py
-----------------------
Unit tests for bootstrap.py.

Tests focus on:
    - Correct output structure (MarginfxResult populated correctly)
    - Reproducibility via seed
    - CI coverage on known DGP
    - Bootstrap distribution properties
"""

import numpy as np
import pytest
from marginfx.bootstrap import bootstrap_ames
from marginfx.core import MarginfxResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def linear_data(rng):
    """
    Known DGP: y = 2*x1 + 3*x2 + noise
    True AMEs: x1 -> 2.0, x2 -> 3.0
    """
    n = 200
    X = rng.standard_normal((n, 2))
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + rng.standard_normal(n) * 0.5
    return X, y


@pytest.fixture
def dummy_model():
    """
    A trivial model object that we can refit.
    Stores coefficients for a linear model manually.
    """
    class LinearModel:
        def __init__(self):
            self.coef_ = np.array([2.0, 3.0])

        def fit(self, X, y):
            # OLS closed form
            self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
            return self

        def predict(self, X):
            return X @ self.coef_

    model = LinearModel()
    return model


@pytest.fixture
def fitted_model(dummy_model, linear_data):
    X, y = linear_data
    return dummy_model.fit(X, y)


@pytest.fixture
def predict_fn():
    def _predict_fn(model, X):
        return model.predict(X)
    return _predict_fn


@pytest.fixture
def fit_fn():
    def _fit_fn(model, X_boot, y_boot):
        import copy
        new_model = copy.deepcopy(model)
        new_model.fit(X_boot, y_boot)
        return new_model
    return _fit_fn


# ---------------------------------------------------------------------------
# Output structure tests
# ---------------------------------------------------------------------------

class TestBootstrapOutput:

    def test_returns_marginfx_result(
        self, fitted_model, linear_data, predict_fn, fit_fn
    ):
        """bootstrap_ames should return a MarginfxResult."""
        X, y = linear_data

        def bound_predict(X_input):
            return predict_fn(fitted_model, X_input)

        result = bootstrap_ames(
            model=fitted_model,
            X=X,
            y=y,
            fit_fn=fit_fn,
            predict_fn=bound_predict,
            n_bootstrap=20,
            seed=42,
            verbose=False,
        )
        assert isinstance(result, MarginfxResult)

    def test_estimates_populated(
        self, fitted_model, linear_data, predict_fn, fit_fn
    ):
        """Result should have estimates dict populated."""
        X, y = linear_data

        def bound_predict(X_input):
            return predict_fn(fitted_model, X_input)

        result = bootstrap_ames(
            model=fitted_model,
            X=X,
            y=y,
            fit_fn=fit_fn,
            predict_fn=bound_predict,
            n_bootstrap=20,
            seed=42,
            verbose=False,
        )
        assert result.estimates is not None
        assert len(result.estimates) == 2

    def test_std_errors_populated(
        self, fitted_model, linear_data, predict_fn, fit_fn
    ):
        """Result should have std_errors dict populated."""
        X, y = linear_data

        def bound_predict(X_input):
            return predict_fn(fitted_model, X_input)

        result = bootstrap_ames(
            model=fitted_model,
            X=X,
            y=y,
            fit_fn=fit_fn,
            predict_fn=bound_predict,
            n_bootstrap=20,
            seed=42,
            verbose=False,
        )
        assert result.std_errors is not None
        assert len(result.std_errors) == 2

    def test_conf_int_populated(
        self, fitted_model, linear_data, predict_fn, fit_fn
    ):
        """Result should have conf_int dict populated."""
        X, y = linear_data

        def bound_predict(X_input):
            return predict_fn(fitted_model, X_input)

        result = bootstrap_ames(
            model=fitted_model,
            X=X,
            y=y,
            fit_fn=fit_fn,
            predict_fn=bound_predict,
            n_bootstrap=20,
            seed=42,
            verbose=False,
        )
        assert result.conf_int is not None
        assert len(result.conf_int) == 2

    def test_conf_int_is_tuple(
        self, fitted_model, linear_data, predict_fn, fit_fn
    ):
        """Each conf_int entry should be a (low, high) tuple."""
        X, y = linear_data

        def bound_predict(X_input):
            return predict_fn(fitted_model, X_input)

        result = bootstrap_ames(
            model=fitted_model,
            X=X,
            y=y,
            fit_fn=fit_fn,
            predict_fn=bound_predict,
            n_bootstrap=20,
            seed=42,
            verbose=False,
        )
        for feature, ci in result.conf_int.items():
            assert isinstance(ci, tuple)
            assert len(ci) == 2
            assert ci[0] < ci[1]

    def test_n_obs_correct(
        self, fitted_model, linear_data, predict_fn, fit_fn
    ):
        """Result n_obs should match input dataset size."""
        X, y = linear_data

        def bound_predict(X_input):
            return predict_fn(fitted_model, X_input)

        result = bootstrap_ames(
            model=fitted_model,
            X=X,
            y=y,
            fit_fn=fit_fn,
            predict_fn=bound_predict,
            n_bootstrap=20,
            seed=42,
            verbose=False,
        )
        assert result.n_obs == X.shape[0]

    def test_n_bootstrap_recorded(
        self, fitted_model, linear_data, predict_fn, fit_fn
    ):
        """Result should record number of bootstrap replicates."""
        X, y = linear_data

        def bound_predict(X_input):
            return predict_fn(fitted_model, X_input)

        result = bootstrap_ames(
            model=fitted_model,
            X=X,
            y=y,
            fit_fn=fit_fn,
            predict_fn=bound_predict,
            n_bootstrap=25,
            seed=42,
            verbose=False,
        )
        assert result.n_bootstrap == 25


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------

class TestReproducibility:

    def _run(self, fitted_model, linear_data, predict_fn, fit_fn, seed):
        X, y = linear_data

        def bound_predict(X_input):
            return predict_fn(fitted_model, X_input)

        return bootstrap_ames(
            model=fitted_model,
            X=X,
            y=y,
            fit_fn=fit_fn,
            predict_fn=bound_predict,
            n_bootstrap=20,
            seed=seed,
            verbose=False,
        )

    def test_same_seed_same_result(
        self, fitted_model, linear_data, predict_fn, fit_fn
    ):
        """Same seed should produce identical results."""
        r1 = self._run(fitted_model, linear_data, predict_fn, fit_fn, seed=42)
        r2 = self._run(fitted_model, linear_data, predict_fn, fit_fn, seed=42)
        for feature in r1.estimates:
            assert abs(r1.estimates[feature] - r2.estimates[feature]) < 1e-10
            assert abs(r1.std_errors[feature] - r2.std_errors[feature]) < 1e-10

    def test_different_seed_different_result(
        self, fitted_model, linear_data, predict_fn, fit_fn
    ):
        """Different seeds should produce different SEs."""
        r1 = self._run(fitted_model, linear_data, predict_fn, fit_fn, seed=42)
        r2 = self._run(fitted_model, linear_data, predict_fn, fit_fn, seed=99)
        # SEs should differ (not guaranteed but overwhelmingly likely)
        ses_differ = any(
            abs(r1.std_errors[f] - r2.std_errors[f]) > 1e-10
            for f in r1.std_errors
        )
        assert ses_differ


# ---------------------------------------------------------------------------
# Statistical validity tests
# ---------------------------------------------------------------------------

class TestStatisticalValidity:

    def test_ame_estimates_close_to_truth(
        self, fitted_model, linear_data, predict_fn, fit_fn
    ):
        """
        AME estimates should be close to true values [2.0, 3.0].
        Uses n=200 dataset so estimates should be reasonably tight.
        """
        X, y = linear_data

        def bound_predict(X_input):
            return predict_fn(fitted_model, X_input)

        result = bootstrap_ames(
            model=fitted_model,
            X=X,
            y=y,
            fit_fn=fit_fn,
            predict_fn=bound_predict,
            n_bootstrap=50,
            seed=42,
            verbose=False,
            feature_names=['x1', 'x2'],
        )
        assert abs(result.estimates['x1'] - 2.0) < 0.1
        assert abs(result.estimates['x2'] - 3.0) < 0.1

    def test_std_errors_positive(
        self, fitted_model, linear_data, predict_fn, fit_fn
    ):
        """All standard errors should be positive."""
        X, y = linear_data

        def bound_predict(X_input):
            return predict_fn(fitted_model, X_input)

        result = bootstrap_ames(
            model=fitted_model,
            X=X,
            y=y,
            fit_fn=fit_fn,
            predict_fn=bound_predict,
            n_bootstrap=50,
            seed=42,
            verbose=False,
        )
        for feature, se in result.std_errors.items():
            assert se > 0, f"SE for {feature} should be positive"

    def test_ci_contains_truth(
        self, fitted_model, linear_data, predict_fn, fit_fn
    ):
        """
        95% CI should contain true AME values [2.0, 3.0].
        This is a single-sample test so not a coverage test,
        but the CI should contain truth for a well-behaved linear DGP.
        """
        X, y = linear_data

        def bound_predict(X_input):
            return predict_fn(fitted_model, X_input)

        result = bootstrap_ames(
            model=fitted_model,
            X=X,
            y=y,
            fit_fn=fit_fn,
            predict_fn=bound_predict,
            n_bootstrap=200,
            seed=42,
            verbose=False,
            feature_names=['x1', 'x2'],
        )
        ci_x1 = result.conf_int['x1']
        ci_x2 = result.conf_int['x2']

        assert ci_x1[0] < 2.0 < ci_x1[1], \
            f"CI {ci_x1} should contain true AME 2.0"
        assert ci_x2[0] < 3.0 < ci_x2[1], \
            f"CI {ci_x2} should contain true AME 3.0"

    def test_alpha_affects_ci_width(
        self, fitted_model, linear_data, predict_fn, fit_fn
    ):
        """Smaller alpha (wider CI) should produce wider intervals."""
        X, y = linear_data

        def bound_predict(X_input):
            return predict_fn(fitted_model, X_input)

        def run(alpha):
            return bootstrap_ames(
                model=fitted_model,
                X=X,
                y=y,
                fit_fn=fit_fn,
                predict_fn=bound_predict,
                n_bootstrap=100,
                alpha=alpha,
                seed=42,
                verbose=False,
                feature_names=['x1', 'x2'],
            )

        r_95 = run(alpha=0.05)
        r_90 = run(alpha=0.10)

        width_95 = r_95.conf_int['x1'][1] - r_95.conf_int['x1'][0]
        width_90 = r_90.conf_int['x1'][1] - r_90.conf_int['x1'][0]

        assert width_95 > width_90, "95% CI should be wider than 90% CI"
