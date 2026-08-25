"""
tests/test_riesz.py
-------------------
Unit tests for riesz.py.

The decisive test is the Riesz representation identity itself (Lemma 1):

    E[ alpha_h(X) g(X) ]  ==  E[ w(X) D_h g(X) ]     for every g in L^1(P)

If the fitted representer satisfies this for a spanning set of g, it is the
right object. Everything else -- mean zero, boundedness, the closed forms --
follows from or corroborates that identity.
"""

import numpy as np
import pytest

from marginfx.core import trimming_weight, window_difference
from marginfx.riesz import (
    KnownRiesz,
    PropensityRiesz,
    SieveRiesz,
    _poly_basis,
    _poly_powers,
    _ridge_logistic,
    gaussian_window_riesz,
)


@pytest.fixture
def rng():
    return np.random.default_rng(7)


@pytest.fixture
def X(rng):
    return rng.standard_normal((4000, 3))


# ---------------------------------------------------------------------------
# Basis
# ---------------------------------------------------------------------------

class TestPolyBasis:

    def test_term_count_degree_two(self):
        # 1 + d + d(d+1)/2 for d = 3  ->  1 + 3 + 6 = 10
        assert len(_poly_powers(3, 2)) == 10

    def test_includes_bias(self):
        assert _poly_powers(2, 1)[0] == ()

    def test_bias_column_is_one(self, X):
        B = _poly_basis(X, _poly_powers(3, 2))
        np.testing.assert_allclose(B[:, 0], 1.0)

    def test_linear_terms_are_the_features(self, X):
        powers = _poly_powers(3, 1)
        B = _poly_basis(X, powers)
        np.testing.assert_allclose(B[:, 1:], X)


# ---------------------------------------------------------------------------
# Sieve Riesz regression
# ---------------------------------------------------------------------------

class TestSieveRiesz:

    @pytest.mark.parametrize("g,name", [
        (lambda Z: Z[:, 0], "x0"),
        (lambda Z: Z[:, 1], "x1"),
        (lambda Z: Z[:, 0] ** 2, "x0^2"),
        (lambda Z: Z[:, 0] * Z[:, 1], "x0*x1"),
        (lambda Z: 3.0 * Z[:, 0] - 2.0 * Z[:, 2], "linear combo"),
    ])
    def test_representation_identity(self, X, g, name):
        h = 0.05
        w = trimming_weight(X, 0, h)
        alpha = SieveRiesz(degree=3, ridge=1e-8).fit(X, 0, h, w).predict(X)

        lhs = np.mean(alpha * g(X))
        rhs = np.mean(w * window_difference(X, 0, g, h))
        assert lhs == pytest.approx(rhs, abs=0.05)

    def test_mean_zero(self, X):
        h = 0.05
        w = trimming_weight(X, 0, h)
        alpha = SieveRiesz(degree=2).fit(X, 0, h, w).predict(X)
        assert abs(alpha.mean()) < 0.05

    def test_targets_the_named_feature(self, X):
        """A representer for coordinate 1 should reproduce D_h on x1, not x0."""
        h = 0.05
        w = trimming_weight(X, 1, h)
        alpha = SieveRiesz(degree=3, ridge=1e-8).fit(X, 1, h, w).predict(X)
        assert np.mean(alpha * X[:, 1]) == pytest.approx(w.mean(), abs=0.05)
        assert abs(np.mean(alpha * X[:, 0])) < 0.05

    def test_truncation_respected(self, X):
        h = 0.05
        w = trimming_weight(X, 0, h)
        est = SieveRiesz(degree=3, alpha_bound=0.5).fit(X, 0, h, w)
        assert np.all(np.abs(est.predict(X)) <= 0.5 + 1e-12)

    def test_handles_constant_feature(self, rng):
        X = np.column_stack([rng.standard_normal(500), np.ones(500)])
        w = np.ones(500)
        alpha = SieveRiesz(degree=2).fit(X, 0, 0.05, w).predict(X)
        assert np.all(np.isfinite(alpha))

    def test_fit_returns_self(self, X):
        est = SieveRiesz()
        assert est.fit(X, 0, 0.05, np.ones(len(X))) is est

    def test_predict_shape(self, X):
        est = SieveRiesz().fit(X, 0, 0.05, np.ones(len(X)))
        assert est.predict(X[:17]).shape == (17,)


# ---------------------------------------------------------------------------
# Known representer
# ---------------------------------------------------------------------------

class TestKnownRiesz:

    def test_passthrough(self, X):
        k = KnownRiesz(lambda Z: Z[:, 0] * 2.0)
        np.testing.assert_allclose(k.predict(X), X[:, 0] * 2.0)

    def test_fit_is_noop(self, X):
        k = KnownRiesz(lambda Z: np.zeros(len(Z)))
        assert k.fit(X, 0, 0.05, None) is k


class TestGaussianRiesz:
    """
    For X ~ N(0, I) the support is unbounded, so w == 1 and

        alpha_h(u) = exp(-h^2/2) sinh(h u_j) / h
    """

    def test_reproduces_derivative_of_identity(self, rng):
        Z = rng.standard_normal((200000, 2))
        a = gaussian_window_riesz(0, 0.05).predict(Z)
        # D_h x0 = 1, so E[alpha * x0] = 1
        assert np.mean(a * Z[:, 0]) == pytest.approx(1.0, abs=0.02)

    def test_mean_zero(self, rng):
        Z = rng.standard_normal((200000, 2))
        a = gaussian_window_riesz(0, 0.05).predict(Z)
        assert abs(a.mean()) < 0.02

    def test_orthogonal_to_other_coordinate(self, rng):
        Z = rng.standard_normal((200000, 2))
        a = gaussian_window_riesz(0, 0.05).predict(Z)
        assert abs(np.mean(a * Z[:, 1])) < 0.02

    def test_matches_sieve_estimate(self, rng):
        """The closed form and the estimated sieve should agree."""
        Z = rng.standard_normal((8000, 2))
        h = 0.1
        w = np.ones(len(Z))
        known = gaussian_window_riesz(0, h).predict(Z)
        sieve = SieveRiesz(degree=3, ridge=1e-8).fit(Z, 0, h, w).predict(Z)
        assert np.corrcoef(known, sieve)[0, 1] > 0.95

    def test_is_odd_in_target_coordinate(self):
        a = gaussian_window_riesz(0, 0.05)
        pos = a.predict(np.array([[1.0, 0.0]]))
        neg = a.predict(np.array([[-1.0, 0.0]]))
        assert pos[0] == pytest.approx(-neg[0])


# ---------------------------------------------------------------------------
# Propensity representer
# ---------------------------------------------------------------------------

class TestRidgeLogistic:

    def test_recovers_known_coefficients(self, rng):
        n = 20000
        Z = rng.standard_normal((n, 2))
        eta = 0.5 + 1.5 * Z[:, 0] - 1.0 * Z[:, 1]
        t = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
        beta = _ridge_logistic(Z, t, ridge=1e-8)
        assert beta[0] == pytest.approx(0.5, abs=0.1)
        assert beta[1] == pytest.approx(1.5, abs=0.1)
        assert beta[2] == pytest.approx(-1.0, abs=0.1)

    def test_handles_separable_data(self):
        Z = np.arange(-10, 10, dtype=float).reshape(-1, 1)
        t = (Z[:, 0] > 0).astype(float)
        beta = _ridge_logistic(Z, t, ridge=1.0)
        assert np.all(np.isfinite(beta))


class TestPropensityRiesz:

    def test_representation_identity_for_contrast(self, rng):
        """
        For a binary x_j the representer must satisfy
            E[alpha(X) g(X)] == E[g(x^(j,1)) - g(x^(j,0))]
        """
        n = 30000
        cont = rng.standard_normal(n)
        # Propensity genuinely depends on the continuous feature.
        e = 1 / (1 + np.exp(-(0.8 * cont)))
        binary = (rng.random(n) < e).astype(float)
        X = np.column_stack([cont, binary])

        est = PropensityRiesz(ridge=1e-6, clip=1e-4).fit(X, 1)
        alpha = est.predict(X)

        def g(Z):
            return 2.0 * Z[:, 1] + 0.5 * Z[:, 0]

        lhs = np.mean(alpha * g(X))
        X1, X0 = X.copy(), X.copy()
        X1[:, 1] = 1.0
        X0[:, 1] = 0.0
        rhs = np.mean(g(X1) - g(X0))
        assert lhs == pytest.approx(rhs, abs=0.15)

    def test_sign_follows_treatment_status(self, rng):
        n = 500
        X = np.column_stack([
            rng.standard_normal(n),
            (rng.random(n) < 0.5).astype(float),
        ])
        alpha = PropensityRiesz().fit(X, 1).predict(X)
        assert np.all(alpha[X[:, 1] == 1.0] > 0)
        assert np.all(alpha[X[:, 1] == 0.0] < 0)

    def test_clipping_bounds_weights(self, rng):
        n = 2000
        cont = rng.standard_normal(n)
        # Near-deterministic assignment would blow up unclipped weights.
        binary = (cont > 0).astype(float)
        X = np.column_stack([cont, binary])
        est = PropensityRiesz(clip=0.05).fit(X, 1)
        assert np.max(np.abs(est.predict(X))) <= 1.0 / 0.05 + 1e-9

    def test_fit_returns_self(self, rng):
        X = np.column_stack([rng.standard_normal(100),
                             (rng.random(100) < 0.5).astype(float)])
        est = PropensityRiesz()
        assert est.fit(X, 1) is est
