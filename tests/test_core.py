"""
tests/test_core.py
------------------
Unit tests for core.py -- the window estimand primitives.

Ground truth strategy
---------------------
For a linear f the centered difference is exact at every h:

    D_h f(x) = beta_j     for all h > 0

so the window AME of a linear model is beta_j * E[w], with the trimming weight
the only thing separating it from the classical coefficient. That identity is
sharp enough to test against directly.

For f(x) = x1^2 the centered difference is also exact:

    (f(x+h) - f(x-h)) / 2h = 2*x1

since the h^2 terms cancel -- a property specific to the centered difference,
and one of the reasons the paper prefers it over a one-sided difference.
"""

import numpy as np
import pandas as pd
import pytest

from marginfx.core import (
    MarginfxResult,
    compute_adaptive_h,
    is_integer_valued,
    contrast,
    plugin_ame,
    plugin_ames,
    pointwise_effects,
    resolve_categorical,
    resolve_h,
    support_bounds,
    trimming_weight,
    window_difference,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def X_linear(rng):
    return rng.standard_normal((200, 2))


@pytest.fixture
def linear_predict_fn():
    """f(x) = 2*x0 + 3*x1 -- D_h f is exactly [2.0, 3.0] at any h."""
    def predict_fn(X):
        return 2.0 * X[:, 0] + 3.0 * X[:, 1]
    return predict_fn


@pytest.fixture
def quadratic_predict_fn():
    """f(x) = x0^2 + x1 -- D_h f for x0 is exactly 2*x0 at any h."""
    def predict_fn(X):
        return X[:, 0] ** 2 + X[:, 1]
    return predict_fn


# ---------------------------------------------------------------------------
# Step size
# ---------------------------------------------------------------------------

class TestAdaptiveH:

    def test_formula(self, rng):
        X = rng.standard_normal((500, 3)) * np.array([1.0, 5.0, 20.0])
        h = compute_adaptive_h(X)
        np.testing.assert_allclose(h, 0.05 * X.std(axis=0))

    def test_floor_applies_to_constant_feature(self):
        X = np.column_stack([np.ones(50), np.arange(50, dtype=float)])
        h = compute_adaptive_h(X)
        assert h[0] == pytest.approx(1e-4)

    def test_integer_floor_applies(self, rng):
        """
        A count feature has no mass between its levels, so the default window
        is widened to the one-unit contrast.
        """
        X = np.column_stack([
            rng.integers(0, 4, size=500).astype(float),
            rng.standard_normal(500),
        ])
        assert 0.05 * X[:, 0].std() < 0.5
        h = compute_adaptive_h(X)
        assert h[0] == pytest.approx(0.5)

    def test_integer_floor_never_lowers_h(self, rng):
        """The floor is a floor: a wide-ranging count keeps 0.05 * std."""
        X = rng.integers(0, 200, size=(500, 1)).astype(float)
        h = compute_adaptive_h(X)
        assert h[0] > 0.5
        assert h[0] == pytest.approx(0.05 * X[:, 0].std())

    def test_no_floor_for_continuous_feature(self, rng):
        """A continuous feature with small spread is left alone."""
        X = (rng.standard_normal((500, 1)) * 0.3)
        h = compute_adaptive_h(X)
        assert h[0] < 0.5
        assert h[0] == pytest.approx(0.05 * X[:, 0].std())

    def test_explicit_h_bypasses_the_floor(self, rng):
        """The floor is part of 'adaptive' only; an explicit h is the caller's."""
        X = rng.integers(0, 4, size=(500, 2)).astype(float)
        np.testing.assert_allclose(resolve_h(X, 0.01), np.full(2, 0.01))
        np.testing.assert_allclose(resolve_h(X, np.array([0.02, 0.3])), [0.02, 0.3])


class TestIsIntegerValued:

    def test_detects_counts(self):
        assert is_integer_valued(np.array([0.0, 1.0, 2.0, 3.0]))

    def test_rejects_continuous(self):
        assert not is_integer_valued(np.array([0.0, 1.5, 2.0]))

    def test_rejects_constant(self):
        """One level is not a count feature; it falls to the 1e-4 floor."""
        assert not is_integer_valued(np.full(10, 3.0))

    def test_handles_negative_integers(self):
        assert is_integer_valued(np.array([-2.0, -1.0, 0.0, 1.0]))

    def test_ignores_non_finite(self):
        assert is_integer_valued(np.array([1.0, 2.0, np.nan]))
        assert not is_integer_valued(np.array([np.nan, np.nan]))

    def test_resolve_scalar(self, X_linear):
        h = resolve_h(X_linear, 0.01)
        np.testing.assert_allclose(h, np.full(2, 0.01))

    def test_resolve_array(self, X_linear):
        h = resolve_h(X_linear, np.array([0.1, 0.2]))
        np.testing.assert_allclose(h, [0.1, 0.2])

    def test_resolve_rejects_bad_string(self, X_linear):
        with pytest.raises(ValueError):
            resolve_h(X_linear, 'auto')

    def test_resolve_rejects_wrong_length(self, X_linear):
        with pytest.raises(ValueError):
            resolve_h(X_linear, np.array([0.1, 0.2, 0.3]))


# ---------------------------------------------------------------------------
# Trimming
# ---------------------------------------------------------------------------

class TestTrimming:

    def test_excludes_boundary_points(self):
        X = np.linspace(0.0, 1.0, 101).reshape(-1, 1)
        w = trimming_weight(X, 0, h=0.1)
        # Points within 0.1 of either end fall outside Omega_{j,h}.
        assert w[0] == 0.0
        assert w[-1] == 0.0
        assert w[50] == 1.0

    def test_weights_are_binary(self, X_linear):
        w = trimming_weight(X_linear, 0, h=0.05)
        assert set(np.unique(w)) <= {0.0, 1.0}

    def test_larger_h_trims_more(self, X_linear):
        small = trimming_weight(X_linear, 0, h=0.01).mean()
        large = trimming_weight(X_linear, 0, h=0.5).mean()
        assert large <= small

    def test_explicit_bounds_are_respected(self, X_linear):
        """
        Trimming must be computable against fixed global bounds, so that a
        subsample does not silently retarget the estimand.
        """
        bounds = support_bounds(X_linear)
        sub = X_linear[:20]
        w_global = trimming_weight(sub, 0, 0.1, bounds)
        w_local = trimming_weight(sub, 0, 0.1)
        # Against the wider global support, no fewer points survive.
        assert w_global.sum() >= w_local.sum()

    def test_support_bounds(self, X_linear):
        lower, upper = support_bounds(X_linear)
        np.testing.assert_allclose(lower, X_linear.min(axis=0))
        np.testing.assert_allclose(upper, X_linear.max(axis=0))


# ---------------------------------------------------------------------------
# Difference operator
# ---------------------------------------------------------------------------

class TestWindowDifference:

    def test_exact_for_linear(self, X_linear, linear_predict_fn):
        for h in [1e-4, 0.05, 1.0]:
            d0 = window_difference(X_linear, 0, linear_predict_fn, h)
            d1 = window_difference(X_linear, 1, linear_predict_fn, h)
            np.testing.assert_allclose(d0, 2.0)
            np.testing.assert_allclose(d1, 3.0)

    def test_centered_difference_exact_for_quadratic(
        self, X_linear, quadratic_predict_fn
    ):
        """The h^2 terms cancel in a centered difference, so this is exact."""
        d = window_difference(X_linear, 0, quadratic_predict_fn, 0.3)
        np.testing.assert_allclose(d, 2.0 * X_linear[:, 0], atol=1e-10)

    def test_does_not_mutate_input(self, X_linear, linear_predict_fn):
        before = X_linear.copy()
        window_difference(X_linear, 0, linear_predict_fn, 0.1)
        np.testing.assert_array_equal(X_linear, before)

    def test_shape(self, X_linear, linear_predict_fn):
        assert window_difference(
            X_linear, 0, linear_predict_fn, 0.1
        ).shape == (200,)


class TestContrast:

    def test_switches_zero_to_one(self):
        X = np.column_stack([np.zeros(10), np.arange(10, dtype=float)])
        c = contrast(X, 0, lambda Z: 5.0 * Z[:, 0])
        np.testing.assert_allclose(c, 5.0)

    def test_holds_other_features_fixed(self):
        X = np.column_stack([np.zeros(5), np.array([1.0, 2, 3, 4, 5])])
        c = contrast(X, 0, lambda Z: Z[:, 0] * Z[:, 1])
        np.testing.assert_allclose(c, X[:, 1])

    def test_does_not_mutate_input(self):
        X = np.column_stack([np.zeros(5), np.ones(5)])
        before = X.copy()
        contrast(X, 0, lambda Z: Z[:, 0])
        np.testing.assert_array_equal(X, before)


# ---------------------------------------------------------------------------
# Plug-in
# ---------------------------------------------------------------------------

class TestPlugin:

    def test_linear_recovers_beta_times_weight(
        self, X_linear, linear_predict_fn
    ):
        h = 0.05
        w = trimming_weight(X_linear, 0, h)
        est = plugin_ame(X_linear, 0, linear_predict_fn, h, weights=w)
        assert est == pytest.approx(2.0 * w.mean())

    def test_not_renormalized_by_weight_mass(self, linear_predict_fn):
        """
        theta_{j,h} = E[w D_h f], not E[w D_h f] / E[w]. Renormalizing would
        target the effect conditional on being untrimmed, a different
        functional.
        """
        X = np.column_stack([np.linspace(0.0, 1.0, 101), np.zeros(101)])
        h = 0.2
        w = trimming_weight(X, 0, h)
        est = plugin_ame(X, 0, linear_predict_fn, h, weights=w)
        assert est == pytest.approx(2.0 * w.mean())
        assert est < 2.0  # strictly shrunk by trimming

    def test_weights_none_means_untrimmed(self, X_linear, linear_predict_fn):
        est = plugin_ame(X_linear, 0, linear_predict_fn, 0.05, weights=None)
        assert est == pytest.approx(2.0)

    def test_pointwise_effects_categorical_ignores_h(self):
        X = np.column_stack([np.zeros(10), np.ones(10)])
        eff = pointwise_effects(
            X, 0, lambda Z: 3.0 * Z[:, 0], h=99.0, is_categorical=True
        )
        np.testing.assert_allclose(eff, 3.0)

    def test_all_features(self, X_linear, linear_predict_fn):
        out = plugin_ames(X_linear, linear_predict_fn, trim=False)
        assert out['x0'] == pytest.approx(2.0)
        assert out['x1'] == pytest.approx(3.0)

    def test_feature_names_used(self, X_linear, linear_predict_fn):
        out = plugin_ames(
            X_linear, linear_predict_fn, feature_names=['a', 'b'], trim=False
        )
        assert set(out) == {'a', 'b'}

    def test_trimming_shrinks_estimate(self, X_linear, linear_predict_fn):
        trimmed = plugin_ames(X_linear, linear_predict_fn, h=0.5, trim=True)
        untrimmed = plugin_ames(X_linear, linear_predict_fn, h=0.5, trim=False)
        assert abs(trimmed['x0']) < abs(untrimmed['x0'])


class TestResolveCategorical:

    def test_by_name(self):
        assert resolve_categorical(['b'], ['a', 'b', 'c']) == {1}

    def test_by_index(self):
        assert resolve_categorical([0, 2], ['a', 'b', 'c']) == {0, 2}

    def test_none_is_empty(self):
        assert resolve_categorical(None, ['a']) == set()

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            resolve_categorical(['zzz'], ['a', 'b'])

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            resolve_categorical([1.5], ['a', 'b'])


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class TestMarginfxResult:

    def test_tidy_minimal(self):
        r = MarginfxResult(estimates={'a': 1.0, 'b': 2.0}, n_obs=10)
        df = r.tidy()
        assert isinstance(df, pd.DataFrame)
        assert list(df['term']) == ['a', 'b']
        assert 'std_error' not in df.columns

    def test_tidy_with_inference(self):
        r = MarginfxResult(
            estimates={'a': 2.0},
            std_errors={'a': 0.5},
            conf_int={'a': (1.0, 3.0)},
            h={'a': 0.05},
            trimmed_fraction={'a': 0.01},
            n_obs=100,
        )
        row = r.tidy().iloc[0]
        assert row['statistic'] == pytest.approx(4.0)
        assert row['p_value'] < 0.001
        assert row['conf_low'] == 1.0
        assert row['h'] == 0.05
        assert row['trimmed'] == 0.01

    def test_zero_se_gives_nan_statistic(self):
        r = MarginfxResult(estimates={'a': 1.0}, std_errors={'a': 0.0},
                           n_obs=5)
        assert np.isnan(r.tidy().iloc[0]['statistic'])

    def test_simultaneous_columns(self):
        r = MarginfxResult(
            estimates={'a': 1.0},
            std_errors={'a': 0.5},
            conf_int={'a': (0.0, 2.0)},
            simultaneous_conf_int={'a': (-0.5, 2.5)},
            n_obs=10,
        )
        row = r.tidy().iloc[0]
        assert row['simul_low'] == -0.5
        assert row['simul_high'] == 2.5

    def test_summary_runs(self, capsys):
        r = MarginfxResult(
            estimates={'a': 1.0}, std_errors={'a': 0.2},
            conf_int={'a': (0.6, 1.4)}, n_obs=50, method='debiased',
            n_folds=5,
        )
        r.summary()
        out = capsys.readouterr().out
        assert 'Window Average Marginal Effects' in out
        assert 'debiased cross-fitted' in out

    def test_summary_warns_for_diagnostic(self, capsys):
        r = MarginfxResult(
            estimates={'a': 1.0}, std_errors={'a': 0.2},
            n_obs=50, method='bootstrap-diagnostic', n_bootstrap=100,
        )
        r.summary()
        assert 'not' in capsys.readouterr().out.lower()

    def test_repr(self):
        r = MarginfxResult(estimates={'a': 1.0}, n_obs=7)
        assert 'MarginfxResult' in repr(r)
