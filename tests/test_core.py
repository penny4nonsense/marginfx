"""
tests/test_core.py
------------------
Unit tests for core.py.

Ground truth validation strategy:
    Use simple known functions where true marginal effects are exact:

    Linear:     f(x) = 2*x1 + 3*x2
                True AMEs: [2.0, 3.0]

    Quadratic:  f(x) = x1^2 + x2
                True ME at point x1: 2*x1
                True AME depends on distribution of x1

    Binary:     f(x) = sigmoid(2*x1 + 3*x2)
                True AMEs are not closed form but finite differences
                should be numerically accurate
"""

import numpy as np
import pandas as pd
import pytest
from marginfx.core import (
    me_at_point,
    marginal_effects,
    ame,
    all_ames,
    MarginfxResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def X_linear(rng):
    """Simple 2-feature dataset for linear function tests."""
    return rng.standard_normal((100, 2))


@pytest.fixture
def linear_predict_fn():
    """f(x) = 2*x1 + 3*x2 — true AMEs are exactly [2.0, 3.0]."""
    def predict_fn(X):
        return 2.0 * X[:, 0] + 3.0 * X[:, 1]
    return predict_fn


@pytest.fixture
def quadratic_predict_fn():
    """f(x) = x1^2 + x2 — true ME at point x1 is 2*x1."""
    def predict_fn(X):
        return X[:, 0] ** 2 + X[:, 1]
    return predict_fn


@pytest.fixture
def binary_predict_fn():
    """f(x) = x1 (binary, values 0 and 1)."""
    def predict_fn(X):
        return X[:, 0]
    return predict_fn


@pytest.fixture
def X_categorical(rng):
    """Dataset with binary first feature."""
    X = rng.standard_normal((100, 2))
    X[:, 0] = rng.integers(0, 2, size=100).astype(float)
    return X


# ---------------------------------------------------------------------------
# me_at_point tests
# ---------------------------------------------------------------------------

class TestMeAtPoint:

    def test_linear_exact(self, linear_predict_fn):
        """For f(x) = 2*x1 + 3*x2, ME at any point w.r.t. x1 is exactly 2."""
        x = np.array([1.0, 1.0])
        result = me_at_point(x, feature_idx=0, predict_fn=linear_predict_fn)
        assert abs(result - 2.0) < 1e-6

    def test_linear_second_feature(self, linear_predict_fn):
        """For f(x) = 2*x1 + 3*x2, ME at any point w.r.t. x2 is exactly 3."""
        x = np.array([1.0, 1.0])
        result = me_at_point(x, feature_idx=1, predict_fn=linear_predict_fn)
        assert abs(result - 3.0) < 1e-6

    def test_quadratic_at_known_point(self, quadratic_predict_fn):
        """For f(x) = x1^2 + x2, ME at x1=2 w.r.t. x1 is 2*2=4."""
        x = np.array([2.0, 0.0])
        result = me_at_point(x, feature_idx=0, predict_fn=quadratic_predict_fn)
        assert abs(result - 4.0) < 1e-4

    def test_quadratic_negative_point(self, quadratic_predict_fn):
        """For f(x) = x1^2 + x2, ME at x1=-3 w.r.t. x1 is 2*(-3)=-6."""
        x = np.array([-3.0, 0.0])
        result = me_at_point(x, feature_idx=0, predict_fn=quadratic_predict_fn)
        assert abs(result - (-6.0)) < 1e-4

    def test_categorical_first_difference(self, binary_predict_fn):
        """For f(x) = x1 (binary), first difference is 1 - 0 = 1."""
        x = np.array([0.5, 1.0])  # current value doesn't matter for categorical
        result = me_at_point(
            x, feature_idx=0,
            predict_fn=binary_predict_fn,
            is_categorical=True,
        )
        assert abs(result - 1.0) < 1e-10

    def test_returns_float(self, linear_predict_fn):
        """me_at_point should always return a Python float."""
        x = np.array([1.0, 1.0])
        result = me_at_point(x, feature_idx=0, predict_fn=linear_predict_fn)
        assert isinstance(result, float)

    def test_h_sensitivity(self, quadratic_predict_fn):
        """Smaller h should give more accurate result for smooth functions."""
        x = np.array([1.0, 0.0])
        true_me = 2.0  # 2 * x1 at x1=1

        result_large_h = me_at_point(
            x, feature_idx=0,
            predict_fn=quadratic_predict_fn,
            h=1e-1,
        )
        result_small_h = me_at_point(
            x, feature_idx=0,
            predict_fn=quadratic_predict_fn,
            h=1e-5,
        )

        assert abs(result_large_h - true_me) < 0.01
        assert abs(result_small_h - true_me) < 0.01

# ---------------------------------------------------------------------------
# marginal_effects tests
# ---------------------------------------------------------------------------

class TestMarginalEffects:

    def test_returns_correct_shape(self, X_linear, linear_predict_fn):
        """marginal_effects should return vector of length n_obs."""
        result = marginal_effects(X_linear, feature_idx=0, predict_fn=linear_predict_fn)
        assert result.shape == (X_linear.shape[0],)

    def test_linear_all_equal(self, X_linear, linear_predict_fn):
        """For linear f, ME is constant across all observations."""
        result = marginal_effects(X_linear, feature_idx=0, predict_fn=linear_predict_fn)
        assert np.allclose(result, 2.0, atol=1e-6)

    def test_quadratic_varies_by_observation(self, X_linear, quadratic_predict_fn):
        """For f(x) = x1^2, ME at point x1 is 2*x1 — varies across observations."""
        result = marginal_effects(X_linear, feature_idx=0, predict_fn=quadratic_predict_fn)
        expected = 2.0 * X_linear[:, 0]
        assert np.allclose(result, expected, atol=1e-4)

    def test_categorical_returns_first_differences(self, X_categorical, binary_predict_fn):
        """Categorical ME should be first difference, not derivative."""
        result = marginal_effects(
            X_categorical,
            feature_idx=0,
            predict_fn=binary_predict_fn,
            is_categorical=True,
        )
        # f(x) = x1, first difference is always 1 - 0 = 1
        assert np.allclose(result, 1.0, atol=1e-10)

    def test_returns_ndarray(self, X_linear, linear_predict_fn):
        """marginal_effects should return numpy array."""
        result = marginal_effects(X_linear, feature_idx=0, predict_fn=linear_predict_fn)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# ame tests
# ---------------------------------------------------------------------------

class TestAme:

    def test_linear_exact(self, X_linear, linear_predict_fn):
        """AME of linear function should be exact coefficient."""
        result_x1 = ame(X_linear, feature_idx=0, predict_fn=linear_predict_fn)
        result_x2 = ame(X_linear, feature_idx=1, predict_fn=linear_predict_fn)
        assert abs(result_x1 - 2.0) < 1e-6
        assert abs(result_x2 - 3.0) < 1e-6

    def test_ame_is_mean_of_marginal_effects(self, X_linear, quadratic_predict_fn):
        """AME should equal mean of marginal_effects vector."""
        me_vec = marginal_effects(X_linear, feature_idx=0, predict_fn=quadratic_predict_fn)
        ame_result = ame(X_linear, feature_idx=0, predict_fn=quadratic_predict_fn)
        assert abs(ame_result - np.mean(me_vec)) < 1e-10

    def test_returns_float(self, X_linear, linear_predict_fn):
        """ame should return a Python float."""
        result = ame(X_linear, feature_idx=0, predict_fn=linear_predict_fn)
        assert isinstance(result, float)

    def test_categorical(self, X_categorical, binary_predict_fn):
        """AME for categorical feature should be mean first difference."""
        result = ame(
            X_categorical,
            feature_idx=0,
            predict_fn=binary_predict_fn,
            is_categorical=True,
        )
        assert abs(result - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# all_ames tests
# ---------------------------------------------------------------------------

class TestAllAmes:

    def test_returns_dict(self, X_linear, linear_predict_fn):
        """all_ames should return a dictionary."""
        result = all_ames(X_linear, predict_fn=linear_predict_fn)
        assert isinstance(result, dict)

    def test_correct_number_of_features(self, X_linear, linear_predict_fn):
        """all_ames should return one entry per feature."""
        result = all_ames(X_linear, predict_fn=linear_predict_fn)
        assert len(result) == X_linear.shape[1]

    def test_default_feature_names(self, X_linear, linear_predict_fn):
        """Default feature names should be x0, x1, ..."""
        result = all_ames(X_linear, predict_fn=linear_predict_fn)
        assert list(result.keys()) == ['x0', 'x1']

    def test_custom_feature_names(self, X_linear, linear_predict_fn):
        """Custom feature names should be used as keys."""
        result = all_ames(
            X_linear,
            predict_fn=linear_predict_fn,
            feature_names=['age', 'income'],
        )
        assert list(result.keys()) == ['age', 'income']

    def test_linear_correct_values(self, X_linear, linear_predict_fn):
        """For f(x) = 2*x1 + 3*x2, all_ames should return {x0: 2.0, x1: 3.0}."""
        result = all_ames(X_linear, predict_fn=linear_predict_fn)
        assert abs(result['x0'] - 2.0) < 1e-6
        assert abs(result['x1'] - 3.0) < 1e-6

    def test_categorical_by_index(self, X_categorical, binary_predict_fn):
        """Categorical features specified by index should use first differences."""
        result = all_ames(
            X_categorical,
            predict_fn=binary_predict_fn,
            categorical_features=[0],
        )
        assert abs(result['x0'] - 1.0) < 1e-10

    def test_categorical_by_name(self, X_categorical, binary_predict_fn):
        """Categorical features specified by name should use first differences."""
        result = all_ames(
            X_categorical,
            predict_fn=binary_predict_fn,
            feature_names=['binary_var', 'continuous_var'],
            categorical_features=['binary_var'],
        )
        assert abs(result['binary_var'] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# MarginfxResult tests
# ---------------------------------------------------------------------------

class TestMarginfxResult:

    @pytest.fixture
    def result_no_bootstrap(self):
        """MarginfxResult with point estimates only."""
        return MarginfxResult(
            estimates={'age': 0.032, 'income': 0.008},
            n_obs=1000,
        )

    @pytest.fixture
    def result_with_bootstrap(self):
        """MarginfxResult with full bootstrap output."""
        return MarginfxResult(
            estimates={'age': 0.032, 'income': 0.008},
            std_errors={'age': 0.004, 'income': 0.001},
            conf_int={'age': (0.024, 0.040), 'income': (0.006, 0.010)},
            n_obs=1000,
            n_bootstrap=200,
            alpha=0.05,
        )

    def test_tidy_returns_dataframe(self, result_no_bootstrap):
        """tidy() should return a pandas DataFrame."""
        df = result_no_bootstrap.tidy()
        assert isinstance(df, pd.DataFrame)

    def test_tidy_has_term_column(self, result_no_bootstrap):
        """tidy() DataFrame should have a 'term' column."""
        df = result_no_bootstrap.tidy()
        assert 'term' in df.columns

    def test_tidy_has_estimate_column(self, result_no_bootstrap):
        """tidy() DataFrame should have an 'estimate' column."""
        df = result_no_bootstrap.tidy()
        assert 'estimate' in df.columns

    def test_tidy_correct_values(self, result_no_bootstrap):
        """tidy() estimates should match input."""
        df = result_no_bootstrap.tidy()
        age_row = df[df['term'] == 'age'].iloc[0]
        assert abs(age_row['estimate'] - 0.032) < 1e-10

    def test_tidy_with_bootstrap_has_se(self, result_with_bootstrap):
        """tidy() with bootstrap should include std_error column."""
        df = result_with_bootstrap.tidy()
        assert 'std_error' in df.columns

    def test_tidy_with_bootstrap_has_ci(self, result_with_bootstrap):
        """tidy() with bootstrap should include conf_low and conf_high."""
        df = result_with_bootstrap.tidy()
        assert 'conf_low' in df.columns
        assert 'conf_high' in df.columns

    def test_tidy_with_bootstrap_has_pvalue(self, result_with_bootstrap):
        """tidy() with bootstrap should include p_value."""
        df = result_with_bootstrap.tidy()
        assert 'p_value' in df.columns

    def test_tidy_correct_number_of_rows(self, result_with_bootstrap):
        """tidy() should have one row per feature."""
        df = result_with_bootstrap.tidy()
        assert len(df) == 2

    def test_conf_int_ordering(self, result_with_bootstrap):
        """conf_low should always be less than conf_high."""
        df = result_with_bootstrap.tidy()
        assert (df['conf_low'] < df['conf_high']).all()

    def test_repr(self, result_with_bootstrap):
        """__repr__ should return a readable string."""
        r = repr(result_with_bootstrap)
        assert 'MarginfxResult' in r
        assert 'features=2' in r

    def test_summary_runs(self, result_with_bootstrap, capsys):
        """summary() should print without error."""
        result_with_bootstrap.summary()
        captured = capsys.readouterr()
        assert 'marginfx' in captured.out.lower()
