"""
riesz.py
--------
Estimation of the Riesz representer of the window average marginal effect.

Background
----------
Lemma 1 of the paper gives an exact dual representation of the estimand. With
q = w * p extended by zero outside Omega, the representer is

    alpha_h(u) = ( q(u - h e_j) - q(u + h e_j) ) / ( 2h p(u) )

and for every g in L^1(P),

    E[ w(X) D_h g(X) ] = E[ alpha_h(X) g(X) ]

The representer is what the debiasing correction in dml.py multiplies the
residual by. Estimating it directly from that formula would require a density
and is not attempted here.

Riesz regression
----------------
Instead we use the loss of Chernozhukov et al. (2022). For any candidate alpha,

    E[alpha(X)^2] - 2 E[w(X) D_h alpha(X)]  =  ||alpha - alpha_h||^2 - ||alpha_h||^2

so the population loss on the left is minimized uniquely at alpha_h. Its sample
analog

    L(alpha) = (1/n) sum_i [ alpha(x_i)^2
                             - 2 w_i (alpha(x_i + h e_j) - alpha(x_i - h e_j)) / (2h) ]

is directly computable: it needs only the ability to evaluate candidates at the
shifted points. At no stage is the density p, let alone a derivative of it,
estimated.

For a linear-in-parameters sieve alpha(x) = b(x)' c this loss is quadratic in c,

    L(c) = c' Bbar c - 2 c' mbar,
    Bbar = (1/n) sum_i b(x_i) b(x_i)',
    mbar = (1/n) sum_i w_i ( b(x_i + h e_j) - b(x_i - h e_j) ) / (2h)

so the minimizer is the closed-form ridge solve (Bbar + lam I) c = mbar. No
iterative optimizer is required.

Truncation
----------
Lemma 1 also bounds ||alpha_h||_inf <= p_max / (h p_min), so any estimate may be
truncated at that bound without degrading its L^2 error. This is what enforces
Assumption 4(c). The bound depends on the unknown density ratio, so it is
exposed as `alpha_bound` rather than guessed at; leaving it None applies no
truncation.
"""

import numpy as np
from itertools import combinations_with_replacement
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Polynomial sieve basis
# ---------------------------------------------------------------------------

def _poly_powers(n_features: int, degree: int) -> list:
    """
    Exponent tuples for a total-degree polynomial basis, including the bias.

    Each entry is a tuple of feature indices to be multiplied together; the
    empty tuple is the constant term.
    """
    powers = [()]
    for deg in range(1, degree + 1):
        powers.extend(combinations_with_replacement(range(n_features), deg))
    return powers


def _poly_basis(Z: np.ndarray, powers: list) -> np.ndarray:
    """Evaluate the polynomial basis defined by `powers` at rows of Z."""
    n = Z.shape[0]
    B = np.empty((n, len(powers)), dtype=float)
    for k, combo in enumerate(powers):
        col = np.ones(n, dtype=float)
        for idx in combo:
            col = col * Z[:, idx]
        B[:, k] = col
    return B


# ---------------------------------------------------------------------------
# Sieve Riesz regression
# ---------------------------------------------------------------------------

class SieveRiesz:
    """
    Riesz regression over a polynomial sieve, solved in closed form.

    Parameters
    ----------
    degree : int
        Total degree of the polynomial basis. Default 2.
    ridge : float
        Ridge penalty on the sieve coefficients. Default 1e-6. Also
        regularizes the normal equations, which are near-singular when the
        basis is rich relative to n.
    alpha_bound : float, optional
        Truncate |alpha_hat| at this value (Assumption 4(c)). The theoretical
        bound is p_max / (h p_min). None applies no truncation.

    Notes
    -----
    Features are standardized before the basis is formed, using moments of the
    training fold. Shifted points are transformed with the same moments, so
    the shift stays consistent between the two terms of the loss.
    """

    def __init__(
        self,
        degree: int = 2,
        ridge: float = 1e-6,
        alpha_bound: Optional[float] = None,
    ):
        self.degree = degree
        self.ridge = ridge
        self.alpha_bound = alpha_bound

    # -- internals ---------------------------------------------------------

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu_) / self.sd_

    def _design(self, X: np.ndarray) -> np.ndarray:
        return _poly_basis(self._standardize(X), self.powers_)

    # -- API ---------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        feature_idx: int,
        h: float,
        weights: np.ndarray,
    ) -> "SieveRiesz":
        """
        Fit the representer on a training fold.

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_train, n_features).
        feature_idx : int
            The coordinate j whose window effect is targeted.
        h : float
            Step size, matching the estimand.
        weights : np.ndarray
            Trimming weights for the training rows, shape (n_train,).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        n, d = X.shape

        self.mu_ = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-12] = 1.0
        self.sd_ = sd
        self.powers_ = _poly_powers(d, self.degree)

        B = self._design(X)

        X_plus = X.copy()
        X_minus = X.copy()
        X_plus[:, feature_idx] += h
        X_minus[:, feature_idx] -= h
        dB = (self._design(X_plus) - self._design(X_minus)) / (2.0 * h)

        Bbar = (B.T @ B) / n
        mbar = (weights[:, None] * dB).mean(axis=0)

        p = Bbar.shape[0]
        try:
            self.coef_ = np.linalg.solve(Bbar + self.ridge * np.eye(p), mbar)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(
                Bbar + self.ridge * np.eye(p), mbar, rcond=None
            )[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the fitted representer.

        Parameters
        ----------
        X : np.ndarray
            Points at which to evaluate, shape (n_obs, n_features).

        Returns
        -------
        np.ndarray
            alpha_hat(x_i), shape (n_obs,), truncated at alpha_bound if set.
        """
        alpha = self._design(np.asarray(X, dtype=float)) @ self.coef_
        if self.alpha_bound is not None:
            alpha = np.clip(alpha, -self.alpha_bound, self.alpha_bound)
        return alpha


# ---------------------------------------------------------------------------
# Known representer
# ---------------------------------------------------------------------------

class KnownRiesz:
    """
    Wrapper for a representer available in closed form.

    In designed or simulated data where the covariate density is known,
    alpha_h is exact and the representer error is identically zero.
    Proposition 2 (double robustness) then applies: valid inference requires
    no rate -- indeed no consistency -- from the learner.

    Parameters
    ----------
    fn : Callable
        alpha(X) -> np.ndarray of shape (n_obs,). Receives the raw feature
        matrix.
    """

    def __init__(self, fn: Callable):
        self.fn = fn

    def fit(self, X, feature_idx, h, weights) -> "KnownRiesz":
        """No-op; the representer is already known."""
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the known representer."""
        return np.asarray(self.fn(np.asarray(X, dtype=float)), dtype=float)


def gaussian_window_riesz(feature_idx: int, h: float) -> KnownRiesz:
    """
    Closed-form representer for independent standard normal covariates.

    With X ~ N(0, I) the support is unbounded, so Omega_{j,h} is all of R^d and
    w == 1 identically. Then q = p and

        alpha_h(u) = ( p(u - h e_j) - p(u + h e_j) ) / ( 2h p(u) )
                   = exp(-h^2 / 2) * sinh(h u_j) / h

    This is the representer the paper's Simulation 3 calls for. Because
    trimming is vacuous in this design, pass trim=False to the estimator so
    that the weight used in the score matches the weight assumed here.

    Parameters
    ----------
    feature_idx : int
        Coordinate j.
    h : float
        Step size.

    Returns
    -------
    KnownRiesz
    """
    def fn(X: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * h * h) * np.sinh(h * X[:, feature_idx]) / h

    return KnownRiesz(fn)


def uniform_window_riesz(
    feature_idx: int,
    h: float,
    lower: float,
    upper: float,
) -> KnownRiesz:
    """
    Closed-form representer for independent uniform covariates on a box.

    With X_j ~ U(lower, upper) the density is constant in the j direction, so
    it cancels from the representer entirely and only the trimming weight
    survives. Writing w(u) = 1{lower + h <= u_j <= upper - h} for the hard
    indicator of Omega_{j,h}, and q = w p,

        alpha_h(u) = ( q(u - h e_j) - q(u + h e_j) ) / ( 2h p(u) )
                   = ( w(u - h e_j) - w(u + h e_j) ) / ( 2h )
                   = ( 1{u_j >= lower + 2h} - 1{u_j <= upper - 2h} ) / ( 2h )

    so alpha_h is the step function taking -1/(2h) on the lower shell
    [lower, lower + 2h), zero on the interior, and +1/(2h) on the upper shell
    (upper - 2h, upper]. It is bounded by 1/(2h), and integrates to zero
    because the two shells have equal probability.

    This is the bounded-support counterpart to gaussian_window_riesz. There the
    density is unbounded below and the representer is unbounded; here both are
    finite, so the design satisfies the density-bounded assumptions directly.
    Trimming is active rather than vacuous, so pass trim=True and supply the
    same bounds used here, or the weight in the score will not match the weight
    assumed by this representer.

    Parameters
    ----------
    feature_idx : int
        Coordinate j.
    h : float
        Step size.
    lower, upper : float
        Support endpoints for coordinate j. Pass the true endpoints of the
        design rather than the sample minimum and maximum, so that the estimand
        does not drift with the sample.

    Returns
    -------
    KnownRiesz
    """
    if not upper - lower > 4.0 * h:
        raise ValueError(
            f'window too wide for the support: need upper - lower > 4h, '
            f'got {upper - lower} <= {4.0 * h}. The two trimming shells would '
            f'overlap and the representer would not be well defined.'
        )

    def fn(X: np.ndarray) -> np.ndarray:
        xj = X[:, feature_idx]
        hi = (xj >= lower + 2.0 * h).astype(float)
        lo = (xj <= upper - 2.0 * h).astype(float)
        return (hi - lo) / (2.0 * h)

    return KnownRiesz(fn)


# ---------------------------------------------------------------------------
# Propensity representer for binary features
# ---------------------------------------------------------------------------

def _ridge_logistic(
    Z: np.ndarray,
    t: np.ndarray,
    ridge: float = 1e-3,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Ridge-penalized logistic regression by IRLS, with intercept.

    Implemented directly on numpy so that the core package does not depend on
    scikit-learn.

    Parameters
    ----------
    Z : np.ndarray
        Design matrix without intercept, shape (n, k).
    t : np.ndarray
        Binary targets in {0, 1}, shape (n,).
    ridge : float
        L2 penalty. The intercept is not penalized.
    max_iter : int
        Maximum IRLS iterations.
    tol : float
        Convergence tolerance on the coefficient update.

    Returns
    -------
    np.ndarray
        Coefficients of shape (k + 1,), intercept first.
    """
    n, k = Z.shape
    D = np.hstack([np.ones((n, 1)), Z])
    beta = np.zeros(k + 1)

    penalty = ridge * np.eye(k + 1)
    penalty[0, 0] = 0.0

    for _ in range(max_iter):
        eta = D @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        s = np.clip(mu * (1.0 - mu), 1e-8, None)
        grad = D.T @ (t - mu) - penalty @ beta
        hess = (D * s[:, None]).T @ D + penalty
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break

    return beta


class PropensityRiesz:
    """
    Representer for a binary feature: inverse propensity weights.

    For a binary x_j the window difference is replaced by the contrast
    f(x^(j,1)) - f(x^(j,0)), and the same Riesz representation delivers

        alpha(x) = 1{x_j = 1} / e(x_-j)  -  1{x_j = 0} / (1 - e(x_-j))

    with e(x_-j) = P(x_j = 1 | x_-j). The resulting orthogonal score is the
    familiar doubly robust / AIPW score for an average treatment effect.

    Parameters
    ----------
    ridge : float
        L2 penalty for the propensity fit. Default 1e-3.
    clip : float
        Propensities are clipped to [clip, 1 - clip] to keep the weights
        bounded, which also enforces Assumption 4(c). Default 0.01.
    """

    def __init__(self, ridge: float = 1e-3, clip: float = 0.01):
        self.ridge = ridge
        self.clip = clip

    def fit(
        self,
        X: np.ndarray,
        feature_idx: int,
        h: float = None,
        weights: np.ndarray = None,
    ) -> "PropensityRiesz":
        """
        Fit the propensity model on a training fold.

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_train, n_features).
        feature_idx : int
            Index of the binary feature.
        h : float
            Unused; present for interface compatibility.
        weights : np.ndarray
            Unused; binary contrasts are not trimmed.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        self.feature_idx_ = feature_idx

        others = [c for c in range(X.shape[1]) if c != feature_idx]
        self.others_ = others

        t = (X[:, feature_idx] > 0.5).astype(float)
        Z = X[:, others]

        self.mu_ = Z.mean(axis=0)
        sd = Z.std(axis=0)
        sd[sd < 1e-12] = 1.0
        self.sd_ = sd

        self.beta_ = _ridge_logistic((Z - self.mu_) / self.sd_, t, self.ridge)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the inverse propensity representer.

        Parameters
        ----------
        X : np.ndarray
            Points at which to evaluate, shape (n_obs, n_features).

        Returns
        -------
        np.ndarray
            alpha(x_i), shape (n_obs,).
        """
        X = np.asarray(X, dtype=float)
        Z = (X[:, self.others_] - self.mu_) / self.sd_
        eta = self.beta_[0] + Z @ self.beta_[1:]
        e = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        e = np.clip(e, self.clip, 1.0 - self.clip)

        xj = (X[:, self.feature_idx_] > 0.5).astype(float)
        return xj / e - (1.0 - xj) / (1.0 - e)
