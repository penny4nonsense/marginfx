"""
tests/test_bootstrap.py
-----------------------
Unit tests for bootstrap.py -- the refitting bootstrap, retained as a
diagnostic only.

The behavioural contracts that matter here are the ones the paper pins down:

- replicates are evaluated at the ORIGINAL sample D, never at D^(b), so a
  replicate depends on the draw only through f_hat^(b);
- h and the trimming weight come from the original sample and are held fixed,
  since they define the estimand;
- the result is labelled a diagnostic, so nobody mistakes the dispersion for
  a valid standard error.
"""

import numpy as np
import pytest

from marginfx.bootstrap import _bootstrap_replicate, bootstrap_diagnostic
from marginfx.core import support_bounds


@pytest.fixture
def rng():
    return np.random.default_rng(3)


@pytest.fixture
def data(rng):
    n = 400
    X = rng.standard_normal((n, 2))
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + rng.standard_normal(n)
    return X, y


class DummyModel:
    """Linear model with fixed coefficients; refits are no-ops."""

    def __init__(self, coefs=(2.0, 3.0)):
        self.coefs = np.asarray(coefs, dtype=float)

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coefs


@pytest.fixture
def dummy():
    model = DummyModel()

    def predict_fn(X):
        return model.predict(X)

    def fit_fn(current, X_boot, y_boot):
        return DummyModel(current.coefs)

    return model, predict_fn, fit_fn


# ---------------------------------------------------------------------------
# Evaluation point
# ---------------------------------------------------------------------------

class TestEvaluationPoint:

    def test_replicate_evaluates_at_original_sample(self, data, dummy, rng):
        """
        The replicate estimate must be computed at X, not at the resampled
        X_boot. With a deterministic refit the answer is then identical to
        the full-sample plug-in on every draw.
        """
        X, y = data
        model, _, fit_fn = dummy
        bounds = support_bounds(X)
        h = np.array([0.05, 0.05])

        seen = []
        for _ in range(5):
            out = _bootstrap_replicate(
                model, X, y, fit_fn, ['x0', 'x1'], None,
                h, True, bounds, rng,
            )
            seen.append(out['x0'])

        # Identical across draws: the only channel from the draw to the
        # estimate is f_hat^(b), which here is constant.
        assert len(set(np.round(seen, 12))) == 1

    def test_replicate_uses_supplied_h_not_recomputed(self, data, dummy, rng):
        X, y = data
        model, _, fit_fn = dummy
        bounds = support_bounds(X)

        big_h = np.array([1.5, 1.5])
        out = _bootstrap_replicate(
            model, X, y, fit_fn, ['x0', 'x1'], None,
            big_h, True, bounds, rng,
        )
        # A large fixed h trims heavily, shrinking the estimate well below 2.
        assert out['x0'] < 2.0


# ---------------------------------------------------------------------------
# Aggregate behaviour
# ---------------------------------------------------------------------------

class TestBootstrapDiagnostic:

    def test_labelled_as_diagnostic(self, data, dummy):
        X, y = data
        model, predict_fn, fit_fn = dummy
        res = bootstrap_diagnostic(
            model, X, y, fit_fn, predict_fn,
            n_bootstrap=5, verbose=False, seed=0,
        )
        assert res.method == 'bootstrap-diagnostic'

    def test_zero_replicates_gives_point_estimates_only(self, data, dummy):
        X, y = data
        model, predict_fn, fit_fn = dummy
        res = bootstrap_diagnostic(
            model, X, y, fit_fn, predict_fn,
            n_bootstrap=0, verbose=False,
        )
        assert res.std_errors is None
        assert res.conf_int is None
        assert res.n_bootstrap == 0
        assert res.estimates['x0'] == pytest.approx(2.0, abs=0.05)

    def test_deterministic_refit_gives_zero_dispersion(self, data, dummy):
        X, y = data
        model, predict_fn, fit_fn = dummy
        res = bootstrap_diagnostic(
            model, X, y, fit_fn, predict_fn,
            n_bootstrap=8, verbose=False, seed=0,
        )
        assert res.std_errors['x0'] == pytest.approx(0.0, abs=1e-12)

    def test_reports_h_and_trimmed_fraction(self, data, dummy):
        X, y = data
        model, predict_fn, fit_fn = dummy
        res = bootstrap_diagnostic(
            model, X, y, fit_fn, predict_fn,
            n_bootstrap=0, verbose=False, h=0.5,
        )
        assert res.h['x0'] == 0.5
        assert res.trimmed_fraction['x0'] > 0.0

    def test_categorical_reported_without_h(self, rng):
        n = 300
        X = np.column_stack([
            rng.standard_normal(n),
            (rng.random(n) < 0.5).astype(float),
        ])
        y = X[:, 0] + 2.0 * X[:, 1]
        model = DummyModel((1.0, 2.0))

        res = bootstrap_diagnostic(
            model, X, y,
            lambda cur, Xb, yb: DummyModel(cur.coefs),
            model.predict,
            feature_names=['cont', 'bin'],
            categorical_features=['bin'],
            n_bootstrap=0, verbose=False,
        )
        assert np.isnan(res.h['bin'])
        assert res.trimmed_fraction['bin'] == 0.0
        assert res.estimates['bin'] == pytest.approx(2.0)

    def test_reproducible_with_seed(self, data):
        X, y = data

        def noisy_fit(current, X_boot, y_boot):
            # Refit genuinely depends on the resampled data.
            coefs = np.linalg.lstsq(X_boot, y_boot, rcond=None)[0]
            return DummyModel(coefs)

        model = DummyModel()
        kwargs = dict(
            fit_fn=noisy_fit, predict_fn=model.predict,
            n_bootstrap=6, verbose=False, seed=123,
        )
        a = bootstrap_diagnostic(model, X, y, **kwargs)
        b = bootstrap_diagnostic(model, X, y, **kwargs)
        assert a.std_errors == b.std_errors

    def test_dispersion_positive_when_refit_varies(self, data):
        X, y = data

        def noisy_fit(current, X_boot, y_boot):
            coefs = np.linalg.lstsq(X_boot, y_boot, rcond=None)[0]
            return DummyModel(coefs)

        model = DummyModel()
        res = bootstrap_diagnostic(
            model, X, y, noisy_fit, model.predict,
            n_bootstrap=25, verbose=False, seed=1,
        )
        assert res.std_errors['x0'] > 0
        lo, hi = res.conf_int['x0']
        assert lo < res.estimates['x0'] < hi

    def test_predict_proba_models_use_probabilities(self, rng):
        class ProbaModel:
            def predict_proba(self, X):
                p = 1 / (1 + np.exp(-np.asarray(X, dtype=float)[:, 0]))
                return np.column_stack([1 - p, p])

            def predict(self, X):
                return (self.predict_proba(X)[:, 1] > 0.5).astype(float)

        n = 300
        X = rng.standard_normal((n, 2))
        y = (rng.random(n) < 0.5).astype(float)
        model = ProbaModel()

        res = bootstrap_diagnostic(
            model, X, y,
            lambda cur, Xb, yb: ProbaModel(),
            lambda Z: model.predict_proba(Z)[:, 1],
            n_bootstrap=3, verbose=False, seed=0,
        )
        # Derivative of the logistic in x0 is positive and bounded by 0.25.
        assert 0.0 < res.estimates['x0'] < 0.25
