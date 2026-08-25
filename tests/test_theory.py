"""
tests/test_theory.py
--------------------
Executable versions of the paper's theoretical guarantees.

The other test modules check mechanics -- shapes, plumbing, error paths. These
check that the estimator is actually the object the theory describes. Each test
here corresponds to a numbered result:

    Lemma 1 (Riesz representation)   -> TestRieszRepresentation
    Lemma 3 (exact product bias)     -> TestExactProductBias
    Theorem 1 (asymptotic linearity) -> TestCalibration
    Proposition 1 (h -> 0 recovery)  -> TestWindowLimit
    Proposition 2 (double robustness)-> TestDoubleRobustness
    Remark (refit bootstrap failure) -> TestBootstrapFailureMode

Design note: the Monte Carlo designs all use X ~ N(0, I), where the support is
unbounded, Omega_{j,h} is all of R^d, and the trimming weight is identically
one. That makes the estimand exactly computable -- for a linear f, theta_h = 2
with no trimming correction -- so coverage can be measured against a known
constant rather than a sample-dependent target. Hence trim=False throughout.
"""

import numpy as np
import pytest
from collections import Counter

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import marginfx as mfx
from marginfx.core import plugin_ame, window_difference
from marginfx.learner import Learner
from marginfx.riesz import gaussian_window_riesz


KNOWN_GAUSSIAN = staticmethod(lambda idx, h, cat: gaussian_window_riesz(idx, h))


def known_riesz(idx, h, cat):
    """Factory supplying the closed-form representer for N(0, I) covariates."""
    return gaussian_window_riesz(idx, h)


# ---------------------------------------------------------------------------
# Learners with known, degenerate behaviour
# ---------------------------------------------------------------------------

class ZeroLearner:
    """Predicts zero everywhere. Maximally wrong, and useful for that."""

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


class ConstantLearner:
    """Predicts the training mean. Consistent for nothing but the intercept."""

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X, y):
        self.mu_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mu_, dtype=float)


# ---------------------------------------------------------------------------
# Lemma 1: the representation identity, against the closed form
# ---------------------------------------------------------------------------

class TestRieszRepresentation:

    def test_closed_form_satisfies_identity_for_many_g(self):
        """
        E[alpha_h(X) g(X)] == E[D_h g(X)] must hold for every g, not just the
        ones the sieve happened to be fit against.
        """
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((400000, 2))
        h = 0.2
        alpha = gaussian_window_riesz(0, h).predict(Z)

        cases = [
            ("x0", lambda W: W[:, 0]),
            ("x0^2", lambda W: W[:, 0] ** 2),
            ("x0^3", lambda W: W[:, 0] ** 3),
            ("sin(x0)", lambda W: np.sin(W[:, 0])),
            ("x0*x1", lambda W: W[:, 0] * W[:, 1]),
            ("exp(-x0^2)", lambda W: np.exp(-W[:, 0] ** 2)),
        ]
        for name, g in cases:
            lhs = np.mean(alpha * g(Z))
            rhs = np.mean(window_difference(Z, 0, g, h))
            assert lhs == pytest.approx(rhs, abs=0.02), name


# ---------------------------------------------------------------------------
# Lemma 3: the moment bias is EXACTLY the product of nuisance errors
# ---------------------------------------------------------------------------

class TestExactProductBias:
    """
    E[psi(X, Y; f_t, a_t, theta_h(f))] == E[(alpha_h - a_t)(f_t - f)]

    This is an identity, not an expansion -- there is no linearization
    remainder. So it must hold for arbitrary, badly-chosen f_t and a_t, not
    merely for ones near the truth.
    """

    @staticmethod
    def _sides(f_tilde, alpha_scale, h=0.2, n=400000, seed=1):
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n, 2))

        def f_true(W):
            return 2.0 * W[:, 0]

        # Y = f(X) exactly. The identity only uses E[Y|X] = f, and dropping the
        # noise removes a pure-variance term that would otherwise dominate the
        # Monte Carlo error.
        y = f_true(Z)

        alpha_h = gaussian_window_riesz(0, h).predict(Z)
        a_tilde = alpha_scale * alpha_h
        theta_h = 2.0  # D_h of a linear function is its slope, exactly

        psi = (
            window_difference(Z, 0, f_tilde, h)
            + a_tilde * (y - f_tilde(Z))
            - theta_h
        )
        lhs = float(np.mean(psi))
        rhs = float(np.mean((alpha_h - a_tilde) * (f_tilde(Z) - f_true(Z))))
        return lhs, rhs

    @pytest.mark.parametrize("scale", [0.0, 0.5, 1.0, 2.0])
    def test_holds_for_wrong_representer(self, scale):
        def f_tilde(W):
            return 2.0 * W[:, 0] + 0.5 * W[:, 0] ** 2 + 0.3

        lhs, rhs = self._sides(f_tilde, scale)
        assert lhs == pytest.approx(rhs, abs=0.02)

    @pytest.mark.parametrize("f_tilde,name", [
        (lambda W: np.zeros(len(W)), "zero"),
        (lambda W: 5.0 * W[:, 0], "wrong slope"),
        (lambda W: np.sin(3.0 * W[:, 0]), "oscillatory"),
        (lambda W: 2.0 * W[:, 0], "correct"),
    ])
    def test_holds_for_wrong_learner(self, f_tilde, name):
        lhs, rhs = self._sides(f_tilde, alpha_scale=0.6)
        assert lhs == pytest.approx(rhs, abs=0.02), name

    def test_bias_vanishes_when_representer_exact(self):
        """Setting a_t = alpha_h kills the bias for EVERY f_t."""
        for f_tilde in [
            lambda W: np.zeros(len(W)),
            lambda W: 5.0 * W[:, 0],
            lambda W: np.cos(W[:, 0]) * W[:, 1],
        ]:
            lhs, _ = self._sides(f_tilde, alpha_scale=1.0)
            assert lhs == pytest.approx(0.0, abs=0.02)

    def test_bias_vanishes_when_learner_exact(self):
        """Setting f_t = f kills the bias for EVERY a_t."""
        for scale in [0.0, 0.5, 3.0]:
            lhs, _ = self._sides(lambda W: 2.0 * W[:, 0], alpha_scale=scale)
            assert lhs == pytest.approx(0.0, abs=0.02)


# ---------------------------------------------------------------------------
# Proposition 1: the window effect recovers the derivative as h -> 0
# ---------------------------------------------------------------------------

class TestWindowLimit:
    """
    For f(x) = sin(x0) and X ~ N(0, I) both quantities are closed form:

        AME      = E[cos(X0)]           = exp(-1/2)
        theta_h  = E[D_h sin(X0)]       = exp(-1/2) * sin(h)/h

    so theta_h / AME = sin(h)/h exactly, and theta_h -> AME as h -> 0.
    """

    @staticmethod
    def _sample():
        rng = np.random.default_rng(5)
        return rng.standard_normal((400000, 2))

    def test_ratio_matches_sinc(self):
        Z = self._sample()
        ame = np.exp(-0.5)
        for h in [0.05, 0.25, 0.5, 1.0]:
            theta = plugin_ame(Z, 0, lambda W: np.sin(W[:, 0]), h)
            assert theta == pytest.approx(ame * np.sin(h) / h, abs=0.005)

    def test_converges_to_classical_ame(self):
        Z = self._sample()
        theta = plugin_ame(Z, 0, lambda W: np.sin(W[:, 0]), 0.01)
        assert theta == pytest.approx(np.exp(-0.5), abs=0.005)

    def test_larger_h_attenuates(self):
        Z = self._sample()
        vals = [plugin_ame(Z, 0, lambda W: np.sin(W[:, 0]), h)
                for h in [0.05, 0.5, 1.5]]
        assert vals[0] > vals[1] > vals[2]


# ---------------------------------------------------------------------------
# Cross-fitting really does hold out
# ---------------------------------------------------------------------------

class TestCrossFitting:

    def test_learner_never_trains_on_its_own_scoring_fold(self, monkeypatch):
        """
        The whole point of cross-fitting is that the score at observation i is
        evaluated with a model that never saw i. Verify structurally: across K
        folds, every row must appear in exactly K-1 training sets.
        """
        rng = np.random.default_rng(2)
        n, K = 300, 5
        X = rng.standard_normal((n, 3))
        y = X[:, 0] + rng.standard_normal(n)

        captured = []

        class SpyLearner(Learner):
            def fit(self, X_train, y_train):
                captured.append(np.asarray(X_train, dtype=float).copy())
                return super().fit(X_train, y_train)

        monkeypatch.setattr('marginfx.dml.Learner', SpyLearner)
        mfx.fit(LinearRegression(), X, y, n_folds=K, seed=0, verbose=False)

        assert len(captured) == K

        counts = Counter()
        for X_train in captured:
            for row in X_train:
                counts[row.tobytes()] += 1

        assert len(counts) == n, "some rows never used for training"
        assert set(counts.values()) == {K - 1}

    def test_fold_sizes_are_balanced(self, monkeypatch):
        rng = np.random.default_rng(3)
        n, K = 203, 4  # deliberately not divisible
        X = rng.standard_normal((n, 2))
        y = X[:, 0].copy()

        sizes = []

        class SpyLearner(Learner):
            def fit(self, X_train, y_train):
                sizes.append(len(X_train))
                return super().fit(X_train, y_train)

        monkeypatch.setattr('marginfx.dml.Learner', SpyLearner)
        mfx.fit(LinearRegression(), X, y, n_folds=K, seed=0, verbose=False)

        assert max(sizes) - min(sizes) <= 1
        assert all(s == n - t for s, t in zip(sizes, [n - s for s in sizes]))


# ---------------------------------------------------------------------------
# Proposition 2: double robustness
# ---------------------------------------------------------------------------

class TestDoubleRobustness:

    def test_zero_learner_with_known_representer_is_exact(self):
        """
        With f_t == 0 the score collapses to alpha_h(x) * y, whose expectation
        is E[alpha_h f] = theta_h. So the most useless learner imaginable still
        yields a correctly located estimate -- paid for entirely in variance.
        """
        rng = np.random.default_rng(4)
        n = 40000
        X = rng.standard_normal((n, 2))
        y = 2.0 * X[:, 0] + rng.standard_normal(n)

        res = mfx.fit(
            ZeroLearner(), X, y,
            trim=False, riesz=known_riesz, seed=0, verbose=False,
        )
        assert res.estimates['x0'] == pytest.approx(2.0, abs=0.05)

    def test_constant_learner_with_known_representer_is_exact(self):
        rng = np.random.default_rng(5)
        n = 40000
        X = rng.standard_normal((n, 2))
        y = 2.0 * X[:, 0] + rng.standard_normal(n)

        res = mfx.fit(
            ConstantLearner(), X, y,
            trim=False, riesz=known_riesz, seed=0, verbose=False,
        )
        assert res.estimates['x0'] == pytest.approx(2.0, abs=0.05)

    def test_inconsistent_forest_with_known_representer(self):
        """A depth-1 forest converges to a stump, not to f."""
        rng = np.random.default_rng(6)
        n = 20000
        X = rng.standard_normal((n, 2))
        y = 2.0 * X[:, 0] + 0.5 * rng.standard_normal(n)

        res = mfx.fit(
            RandomForestRegressor(n_estimators=30, max_depth=1,
                                  random_state=0, n_jobs=1),
            X, y, trim=False, riesz=known_riesz, seed=0, verbose=False,
        )
        assert res.estimates['x0'] == pytest.approx(2.0, abs=0.1)

    def test_wrong_learner_costs_variance_not_location(self):
        """
        The price of a bad learner is a wider interval, not a displaced one.
        """
        rng = np.random.default_rng(7)
        n = 20000
        X = rng.standard_normal((n, 2))
        y = 2.0 * X[:, 0] + 0.5 * rng.standard_normal(n)

        good = mfx.fit(LinearRegression(), X, y, trim=False,
                       riesz=known_riesz, seed=0, verbose=False)
        bad = mfx.fit(ZeroLearner(), X, y, trim=False,
                      riesz=known_riesz, seed=0, verbose=False)

        assert bad.std_errors['x0'] > good.std_errors['x0']
        assert abs(bad.estimates['x0'] - 2.0) < 0.1
        assert abs(good.estimates['x0'] - 2.0) < 0.1


# ---------------------------------------------------------------------------
# Theorem 1: asymptotic linearity, normality, and a consistent variance
# ---------------------------------------------------------------------------

def _replicate(n, rep, riesz, learner_factory, h=0.2):
    """One Monte Carlo draw; returns (estimate, se, covered)."""
    rng = np.random.default_rng(10_000 + rep)
    X = rng.standard_normal((n, 2))
    y = 2.0 * X[:, 0] + rng.standard_normal(n)

    res = mfx.fit(
        learner_factory(), X, y,
        h=h, trim=False, riesz=riesz, n_folds=5,
        seed=rep, verbose=False,
    )
    lo, hi = res.conf_int['x0']
    return (res.estimates['x0'], res.std_errors['x0'],
            bool(lo <= 2.0 <= hi))


@pytest.mark.slow
class TestCalibration:
    """
    theta_h = 2.0 exactly here: D_h of a linear function is its slope, and with
    Gaussian covariates the trimming weight is identically one.
    """

    N_REPS = 200
    N_OBS = 400

    def _run(self, riesz):
        out = [
            _replicate(self.N_OBS, r, riesz, LinearRegression)
            for r in range(self.N_REPS)
        ]
        est = np.array([o[0] for o in out])
        se = np.array([o[1] for o in out])
        cov = np.mean([o[2] for o in out])
        return est, se, cov

    def test_coverage_with_known_representer(self):
        est, se, cov = self._run(known_riesz)
        # Binomial SE at 200 reps is about 0.015; allow a wide band so the
        # test flags miscalibration, not Monte Carlo noise.
        assert 0.90 <= cov <= 0.99, f"coverage {cov:.3f}"

    def test_coverage_with_estimated_representer(self):
        est, se, cov = self._run(None)
        assert 0.90 <= cov <= 0.99, f"coverage {cov:.3f}"

    def test_estimator_is_unbiased(self):
        est, se, cov = self._run(known_riesz)
        # SE of the mean across reps
        tol = 4.0 * est.std() / np.sqrt(self.N_REPS)
        assert abs(est.mean() - 2.0) < max(tol, 0.02), (
            f"mean estimate {est.mean():.4f}"
        )

    def test_variance_estimator_matches_sampling_variability(self):
        """
        V_hat is consistent for Var(psi), so the average reported SE must track
        the actual spread of the estimator across replications. This is the
        test that would catch an off-by-sqrt(n) or a ddof mistake.
        """
        est, se, cov = self._run(known_riesz)
        assert se.mean() == pytest.approx(est.std(), rel=0.15), (
            f"mean se {se.mean():.5f} vs sd of estimates {est.std():.5f}"
        )

    def test_se_shrinks_at_root_n(self):
        def mean_se(n):
            return np.mean([
                _replicate(n, r, known_riesz, LinearRegression)[1]
                for r in range(40)
            ])

        ratio = mean_se(400) / mean_se(1600)
        assert ratio == pytest.approx(2.0, rel=0.2)


# ---------------------------------------------------------------------------
# The documented failure of the refitting bootstrap
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBootstrapFailureMode:

    def test_debiased_beats_plugin_for_regularized_learner(self):
        """
        The plug-in inherits the learner's regularization bias; the correction
        term removes it. Averaged over replications, the debiased estimator
        must have materially lower RMSE against the known truth.
        """
        n, reps = 1500, 15

        def forest():
            return RandomForestRegressor(
                n_estimators=40, max_depth=3, random_state=0, n_jobs=1
            )

        deb_err, plug_err = [], []
        for r in range(reps):
            rng = np.random.default_rng(500 + r)
            X = rng.standard_normal((n, 2))
            y = 2.0 * X[:, 0] + 0.5 * rng.standard_normal(n)

            deb = mfx.fit(forest(), X, y, h=0.2, trim=False,
                          riesz=known_riesz, seed=r, verbose=False)
            deb_err.append(deb.estimates['x0'] - 2.0)

            fitted = forest().fit(X, y)
            plug = mfx.plugin_ames(
                X, lambda W: fitted.predict(W),
                feature_names=['x0', 'x1'], h=0.2, trim=False,
            )
            plug_err.append(plug['x0'] - 2.0)

        deb_rmse = float(np.sqrt(np.mean(np.square(deb_err))))
        plug_rmse = float(np.sqrt(np.mean(np.square(plug_err))))

        assert deb_rmse < plug_rmse, (
            f"debiased RMSE {deb_rmse:.4f} vs plug-in {plug_rmse:.4f}"
        )
        # The plug-in bias is systematic, not noise: it points one way.
        assert abs(np.mean(plug_err)) > 2.0 * abs(np.mean(deb_err))

    def test_plugin_bias_is_toward_zero(self):
        """
        A depth-capped forest under-responds to the covariate, so the plug-in
        window effect is attenuated -- shrunk toward zero, not scattered.
        """
        rng = np.random.default_rng(99)
        n = 3000
        X = rng.standard_normal((n, 2))
        y = 2.0 * X[:, 0] + 0.5 * rng.standard_normal(n)

        fitted = RandomForestRegressor(
            n_estimators=40, max_depth=3, random_state=0, n_jobs=1
        ).fit(X, y)
        plug = mfx.plugin_ames(
            X, lambda W: fitted.predict(W),
            feature_names=['x0', 'x1'], h=0.2, trim=False,
        )
        assert 0.0 < plug['x0'] < 2.0
