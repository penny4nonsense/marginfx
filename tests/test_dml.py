"""
tests/test_dml.py
-----------------
Unit tests for dml.py -- the debiased cross-fitted estimator.

Ground truth strategy
---------------------
For a linear DGP fitted with LinearRegression the window AME is beta_j * E[w],
recoverable to three decimals. For a deliberately over-regularized learner
(a depth-capped forest) the plug-in is badly biased while the debiased
estimator is not -- which is the paper's central claim, and is tested directly.

The double robustness test is the sharpest: with a known representer and a
learner that is not even consistent, the estimator must still land on the
truth, because the moment bias vanishes identically rather than
asymptotically.
"""

import numpy as np
import pytest

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import marginfx as mfx
from marginfx.core import support_bounds, trimming_weight
from marginfx.dml import _kfold_indices, _make_representer, crossfit_ames
from marginfx.riesz import (
    PropensityRiesz,
    SieveRiesz,
    gaussian_window_riesz,
)


@pytest.fixture
def rng():
    return np.random.default_rng(11)


@pytest.fixture
def linear_data(rng):
    n = 3000
    X = rng.standard_normal((n, 4))
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + rng.standard_normal(n)
    return X, y


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

class TestKFold:

    def test_partitions_exactly(self, rng):
        folds = _kfold_indices(100, 5, rng)
        test_all = np.concatenate([te for _, te in folds])
        assert len(test_all) == 100
        assert set(test_all) == set(range(100))

    def test_train_and_test_disjoint(self, rng):
        for tr, te in _kfold_indices(97, 4, rng):
            assert not (set(tr) & set(te))
            assert len(tr) + len(te) == 97

    def test_rejects_too_few_folds(self, rng):
        with pytest.raises(ValueError):
            _kfold_indices(50, 1, rng)

    def test_rejects_more_folds_than_rows(self, rng):
        with pytest.raises(ValueError):
            _kfold_indices(3, 5, rng)


# ---------------------------------------------------------------------------
# Representer resolution
# ---------------------------------------------------------------------------

class TestRepresenterResolution:

    def test_default_continuous_is_sieve(self):
        r = _make_representer(None, 'x0', 0, 0.05, False, 2, 1e-6, None)
        assert isinstance(r, SieveRiesz)

    def test_default_categorical_is_propensity(self):
        r = _make_representer(None, 'b', 1, 0.05, True, 2, 1e-6, None)
        assert isinstance(r, PropensityRiesz)

    def test_factory_is_called(self):
        r = _make_representer(
            lambda idx, h, cat: gaussian_window_riesz(idx, h),
            'x0', 0, 0.05, False, 2, 1e-6, None,
        )
        assert hasattr(r, 'predict')

    def test_dict_lookup_is_copied(self):
        shared = SieveRiesz(degree=1)
        a = _make_representer({'x0': shared}, 'x0', 0, 0.05, False,
                              2, 1e-6, None)
        b = _make_representer({'x0': shared}, 'x0', 0, 0.05, False,
                              2, 1e-6, None)
        # Fresh instance per fold, else the last fold's fit leaks.
        assert a is not b
        assert a is not shared

    def test_dict_falls_back_to_default(self):
        r = _make_representer({'other': SieveRiesz()}, 'x0', 0, 0.05, False,
                              2, 1e-6, None)
        assert isinstance(r, SieveRiesz)

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            _make_representer(42, 'x0', 0, 0.05, False, 2, 1e-6, None)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

class TestRecovery:

    def test_linear_dgp(self, linear_data):
        X, y = linear_data
        res = mfx.fit(LinearRegression(), X, y, seed=0, verbose=False)

        bounds = support_bounds(X)
        h = res.h
        for j, (name, beta) in enumerate(
            [('x0', 2.0), ('x1', 3.0), ('x2', 0.0), ('x3', 0.0)]
        ):
            w = trimming_weight(X, j, h[name], bounds)
            assert res.estimates[name] == pytest.approx(
                beta * w.mean(), abs=0.05
            )

    def test_logistic_ame(self, rng):
        n = 4000
        X = rng.standard_normal((n, 3))
        eta = 1.5 * X[:, 0] - 1.0 * X[:, 1]
        p = 1 / (1 + np.exp(-eta))
        y = (rng.random(n) < p).astype(float)

        res = mfx.fit(LogisticRegression(), X, y, seed=0, verbose=False)
        scale = np.mean(p * (1 - p))
        assert res.estimates['x0'] == pytest.approx(1.5 * scale, abs=0.03)
        assert res.estimates['x1'] == pytest.approx(-1.0 * scale, abs=0.03)

    def test_binary_contrast(self, rng):
        n = 3000
        X = np.column_stack([
            rng.standard_normal(n),
            (rng.random(n) < 0.5).astype(float),
        ])
        y = 1.0 * X[:, 0] + 2.0 * X[:, 1] + 0.5 * rng.standard_normal(n)
        res = mfx.fit(
            LinearRegression(), X, y,
            feature_names=['cont', 'bin'], categorical_features=['bin'],
            seed=0, verbose=False,
        )
        assert res.estimates['bin'] == pytest.approx(2.0, abs=0.06)
        assert np.isnan(res.h['bin'])
        assert res.trimmed_fraction['bin'] == 0.0


class TestDebiasing:

    def test_beats_plugin_for_regularized_learner(self, rng):
        """
        A depth-capped forest is heavily regularized, so the plug-in average
        inherits a large bias. The correction term is built to remove exactly
        that, at first order.
        """
        n = 3000
        X = rng.standard_normal((n, 3))
        y = 2.0 * X[:, 0] + X[:, 1] + 0.5 * rng.standard_normal(n)

        def forest():
            return RandomForestRegressor(
                n_estimators=60, max_depth=4, random_state=0, n_jobs=1
            )

        debiased = mfx.fit(
            forest(), X, y, trim=False, seed=1, verbose=False,
            riesz=lambda idx, h, cat: gaussian_window_riesz(idx, h),
        )
        fitted = forest().fit(X, y)
        plugin = mfx.plugin_ames(
            X, lambda Z: fitted.predict(Z),
            feature_names=['x0', 'x1', 'x2'], trim=False,
        )

        assert abs(debiased.estimates['x0'] - 2.0) < abs(plugin['x0'] - 2.0)
        assert abs(debiased.estimates['x0'] - 2.0) < 0.1

    def test_double_robustness_with_known_representer(self, rng):
        """
        Proposition 2: with alpha_h known exactly the moment bias vanishes for
        EVERY candidate f, so an inconsistent learner still yields a correctly
        located estimate. A depth-1 forest converges to a stump, not to f.
        """
        n = 4000
        X = rng.standard_normal((n, 2))
        y = 2.0 * X[:, 0] + 0.5 * rng.standard_normal(n)

        res = mfx.fit(
            RandomForestRegressor(n_estimators=40, max_depth=1,
                                  random_state=0, n_jobs=1),
            X, y, trim=False, seed=2, verbose=False,
            riesz=lambda idx, h, cat: gaussian_window_riesz(idx, h),
        )
        assert res.estimates['x0'] == pytest.approx(2.0, abs=0.15)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class TestInference:

    def test_se_is_influence_sd_over_sqrt_n(self, linear_data):
        X, y = linear_data
        res = mfx.fit(LinearRegression(), X, y, seed=0, verbose=False)
        psi = res.influence['x0']
        expected = psi.std() / np.sqrt(len(psi))
        assert res.std_errors['x0'] == pytest.approx(expected, rel=1e-12)

    def test_point_estimate_is_influence_mean(self, linear_data):
        X, y = linear_data
        res = mfx.fit(LinearRegression(), X, y, seed=0, verbose=False)
        assert res.estimates['x0'] == pytest.approx(
            res.influence['x0'].mean(), rel=1e-12
        )

    def test_noise_features_not_significant(self, linear_data):
        X, y = linear_data
        res = mfx.fit(LinearRegression(), X, y, seed=0, verbose=False)
        for name in ['x2', 'x3']:
            t = res.estimates[name] / res.std_errors[name]
            assert abs(t) < 3.0

    def test_signal_features_significant(self, linear_data):
        X, y = linear_data
        res = mfx.fit(LinearRegression(), X, y, seed=0, verbose=False)
        assert abs(res.estimates['x0'] / res.std_errors['x0']) > 10

    def test_confidence_interval_brackets_estimate(self, linear_data):
        X, y = linear_data
        res = mfx.fit(LinearRegression(), X, y, seed=0, verbose=False)
        for name, est in res.estimates.items():
            lo, hi = res.conf_int[name]
            assert lo < est < hi

    def test_se_shrinks_with_n(self, rng):
        def se_at(n):
            X = rng.standard_normal((n, 2))
            y = 2.0 * X[:, 0] + rng.standard_normal(n)
            return mfx.fit(
                LinearRegression(), X, y, seed=0, verbose=False
            ).std_errors['x0']

        assert se_at(4000) < se_at(500)

    def test_no_multiplier_by_default(self, linear_data):
        X, y = linear_data
        res = mfx.fit(LinearRegression(), X, y, seed=0, verbose=False)
        assert res.simultaneous_conf_int is None

    def test_multiplier_band_wider_than_pointwise(self, linear_data):
        X, y = linear_data
        res = mfx.fit(LinearRegression(), X, y, seed=0, verbose=False,
                      n_multiplier=500)
        for name in res.estimates:
            simul = res.simultaneous_conf_int[name]
            point = res.conf_int[name]
            assert (simul[1] - simul[0]) >= (point[1] - point[0])

    def test_alpha_widens_interval(self, linear_data):
        X, y = linear_data
        narrow = mfx.fit(LinearRegression(), X, y, seed=0, verbose=False,
                         alpha=0.10).conf_int['x0']
        wide = mfx.fit(LinearRegression(), X, y, seed=0, verbose=False,
                       alpha=0.01).conf_int['x0']
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


# ---------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------

class TestCountFeatures:
    """
    The integer floor on adaptive h exists for this case: a piecewise
    constant learner differenced over a sub-unit window on a count feature.
    """

    @pytest.fixture
    def count_data(self, rng):
        n = 3000
        beds = rng.integers(1, 6, size=n).astype(float)
        area = rng.standard_normal(n) * 500 + 1500
        X = np.column_stack([area, beds])
        y = 100.0 * area - 20000.0 * beds + rng.standard_normal(n) * 10000
        return X, y

    def test_adaptive_h_widens_to_one_unit(self, count_data):
        X, y = count_data
        res = mfx.fit(RandomForestRegressor(n_estimators=50, max_depth=6,
                                            random_state=0),
                      X, y, feature_names=['area', 'beds'], n_folds=3,
                      seed=0, verbose=False)
        assert res.h['beds'] == pytest.approx(0.5)
        assert res.h['area'] == pytest.approx(0.05 * X[:, 0].std())

    def test_sub_unit_window_loses_the_effect(self, count_data):
        """
        Over a window narrower than the gap between levels the learner is
        constant almost everywhere, so the difference carries no signal: the
        estimate collapses toward zero. Widening to the one-unit contrast
        recovers the true -20000 to within a factor of two.
        """
        X, y = count_data
        learner = RandomForestRegressor(n_estimators=100, max_depth=8,
                                        random_state=0)
        floored = mfx.fit(learner, X, y, feature_names=['area', 'beds'],
                          n_folds=3, seed=0, verbose=False)
        narrow = mfx.fit(learner, X, y, feature_names=['area', 'beds'],
                         h=np.array([0.05 * X[:, 0].std(),
                                     0.05 * X[:, 1].std()]),
                         n_folds=3, seed=0, verbose=False)

        assert -40000.0 < floored.estimates['beds'] < -10000.0
        assert abs(narrow.estimates['beds']) < 0.1 * abs(
            floored.estimates['beds']
        )

    def test_continuous_feature_is_unaffected(self, count_data):
        """The floor touches counts only; area keeps its 0.05 * std window."""
        X, y = count_data
        res = mfx.fit(RandomForestRegressor(n_estimators=100, max_depth=8,
                                            random_state=0),
                      X, y, feature_names=['area', 'beds'], n_folds=3,
                      seed=0, verbose=False)
        assert 50.0 < res.estimates['area'] < 150.0


class TestMechanics:

    def test_every_observation_is_scored(self, linear_data):
        X, y = linear_data
        res = mfx.fit(LinearRegression(), X, y, seed=0, verbose=False)
        for psi in res.influence.values():
            assert psi.shape == (len(X),)
            assert np.all(np.isfinite(psi))

    def test_reproducible_with_seed(self, linear_data):
        X, y = linear_data
        a = mfx.fit(LinearRegression(), X, y, seed=99, verbose=False)
        b = mfx.fit(LinearRegression(), X, y, seed=99, verbose=False)
        assert a.estimates == b.estimates

    def test_h_held_fixed_across_folds(self, linear_data):
        """h defines the estimand, so it must not drift with the fold."""
        X, y = linear_data
        res = mfx.fit(LinearRegression(), X, y, seed=0, verbose=False)
        expected = 0.05 * X[:, 0].std()
        assert res.h['x0'] == pytest.approx(expected)

    def test_n_folds_recorded(self, linear_data):
        X, y = linear_data
        res = mfx.fit(LinearRegression(), X, y, n_folds=3, seed=0,
                      verbose=False)
        assert res.n_folds == 3
        assert res.method == 'debiased'

    def test_trim_flag_changes_estimand(self, linear_data):
        X, y = linear_data
        trimmed = mfx.fit(LinearRegression(), X, y, h=0.5, seed=0,
                          verbose=False, trim=True)
        untrimmed = mfx.fit(LinearRegression(), X, y, h=0.5, seed=0,
                            verbose=False, trim=False)
        assert trimmed.trimmed_fraction['x0'] > 0
        assert untrimmed.trimmed_fraction['x0'] == 0.0
        assert abs(trimmed.estimates['x0']) < abs(untrimmed.estimates['x0'])

    def test_trimmed_estimate_equals_beta_times_weight_mass(self, rng):
        """
        theta_{j,h} = E[w D_h f], so for a linear f it is exactly
        beta_j * E[w]. Uniform covariates make the trimmed shell large enough
        -- here roughly 40% of the sample -- that the weight cannot quietly go
        missing from the score without this test noticing.
        """
        n, h = 4000, 0.2
        X = rng.random((n, 2))
        y = 2.0 * X[:, 0] + 0.3 * rng.standard_normal(n)

        res = mfx.fit(LinearRegression(), X, y, h=h, trim=True,
                      seed=0, verbose=False)
        w = trimming_weight(X, 0, h, support_bounds(X))

        assert w.mean() < 0.7, "design should trim substantially"
        assert res.estimates['x0'] == pytest.approx(2.0 * w.mean(), abs=0.05)
        # And the untrimmed estimand is the undiscounted slope.
        untrimmed = mfx.fit(LinearRegression(), X, y, h=h, trim=False,
                            seed=0, verbose=False)
        assert untrimmed.estimates['x0'] == pytest.approx(2.0, abs=0.05)

    def test_does_not_mutate_the_estimator(self, linear_data):
        X, y = linear_data
        est = LinearRegression()
        mfx.fit(est, X, y, seed=0, verbose=False)
        assert not hasattr(est, 'coef_')

    def test_dataframe_input_infers_names(self, linear_data):
        pd = pytest.importorskip("pandas")
        X, y = linear_data
        df = pd.DataFrame(X, columns=['a', 'b', 'c', 'd'])
        res = mfx.fit(LinearRegression(), df, y, seed=0, verbose=False)
        assert set(res.estimates) == {'a', 'b', 'c', 'd'}

    def test_length_mismatch_raises(self, linear_data):
        X, y = linear_data
        with pytest.raises(ValueError):
            crossfit_ames(LinearRegression(), X, y[:-5], verbose=False)

    def test_bad_feature_names_length_raises(self, linear_data):
        X, y = linear_data
        with pytest.raises(ValueError):
            crossfit_ames(LinearRegression(), X, y,
                          feature_names=['a', 'b'], verbose=False)

    def test_rejects_fitted_keras_or_torch_instance(self):
        """
        Cross-fitting must retrain per fold; a trained network instance would
        leak the held-out fold into its own score.
        """
        torch = pytest.importorskip("torch")
        model = torch.nn.Linear(3, 1)
        with pytest.raises(TypeError, match="zero-argument callable"):
            mfx.fit(model, np.zeros((10, 3)), np.zeros(10), verbose=False)
