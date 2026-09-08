# Changelog

## 0.3.0 — unreleased

This release changes what `mfx.fit` computes and fixes a defect that produced
silently wrong standard errors for tree ensembles. **Read the two headline
items before upgrading**: existing code will not run unchanged, and results
produced with 0.1.x should be rechecked.

### Fixed: bootstrap standard errors were near zero for random forests

Bootstrap replicates were warm-started from the full-sample fit rather than
refit from scratch. For a scikit-learn random forest this made the refit a
no-op — warm-starting without increasing `n_estimators` adds no trees and
keeps the existing ones — so every replicate returned essentially the original
model and the replicate spread collapsed. The reported standard error was
roughly thirty times too small, and it failed silently: no warning, no error,
just implausibly tight intervals and significance everywhere.

The same mechanism affected other learners less severely:

| learner | 0.1.x behaviour | effect on the standard error |
|---|---|---|
| random forest | never actually refit | ~30x too small |
| gradient boosting / XGBoost | replicate continued training to double the boosting rounds | too small, varying by design |
| Keras | replicate rebuilt from config, then the original weights copied back in | too small |
| Keras, regression with a rescaling wrapper | rebuilt from `get_config()`, which returns the *inner* network's config and drops the wrapper, so replicates were spread on the standardized scale | ~300x too small |
| PyTorch | replicate resumed from the fitted parameters | too small |

All five are fixed: every replicate is now built from a fresh initialization.
The test suite asserts the new behaviour — a refit that leaves parameters
essentially unchanged after one epoch now fails, where before it passed.

**If you published standard errors from 0.1.x for anything other than a linear
or logistic model, recompute them.** The point estimates are unaffected.

### Changed: `mfx.fit` is now a debiased, cross-fitted estimator

`fit` previously took an already-fitted model and returned plug-in average
marginal effects with bootstrap standard errors. It now takes an **unfitted**
learner, fits it once per cross-fitting fold, and returns a debiased estimate
built from an orthogonal score.

```python
# 0.1.x
model = RandomForestRegressor().fit(X, y)
result = mfx.fit(model, X, y, n_bootstrap=200)

# 0.3.0 — same estimator as before
model = RandomForestRegressor().fit(X, y)
result = mfx.bootstrap_diagnostic(model, X, y, n_bootstrap=200)

# 0.3.0 — the debiased estimator, and the new default
result = mfx.fit(RandomForestRegressor(), X, y, n_folds=5)
```

`fit` no longer accepts `n_bootstrap`, and passing an already-fitted model to
it will refit that learner on every fold rather than reusing the fit. The old
behaviour is preserved verbatim under `bootstrap_diagnostic`.

The reason for the change is that the plug-in average inherits the learner's
regularization bias, and resampling cannot see that bias: every replicate
recentres at a similarly regularized fit. Where the bias is small relative to
the standard error this does not matter, and the two estimators agree. Where it
is not — a depth-capped forest at large `n` is the standard case — the
plug-in's intervals shrink around a displaced centre and coverage degrades
without any diagnostic firing.

### Changed: a trimming weight is applied by default

`fit` and `bootstrap_diagnostic` take `trim=True` by default. Observations
within `h` of the boundary of a feature's observed support receive weight zero,
because evaluating at `x ± h` there requires the learner to extrapolate beyond
the region where the regression function is identified.

This changes reported effects for features with mass at a support boundary,
sometimes substantially: a variable that is zero for ninety percent of the
sample, with zero as its observed minimum, now reports an effect on the
untrimmed subpopulation rather than on the full sample. Pass `trim=False` to
recover the previous behaviour, and check which features are affected before
you do.

### Added

- `bootstrap_diagnostic` — the 0.1.x estimator, unchanged in method and with
  the refit defect fixed.
- `bounds` on `fit` and `crossfit_ames` — explicit support bounds for the
  trimming weight, defaulting to the sample extremes as before. Pass the true
  support when it is known: the sample minimum of a bounded covariate sits
  strictly inside its support and moves with `n`, so trimming at the sample
  extremes makes the estimand a different target at every sample size.
- Riesz representer estimation: `SieveRiesz`, `KnownRiesz`, `PropensityRiesz`.
- `gaussian_window_riesz` — the closed-form representer for independent
  standard normal covariates, `exp(-h²/2)·sinh(h·u_j)/h`.
- `uniform_window_riesz` — the closed-form representer for independent uniform
  covariates on a box, a bounded step function taking ±1/(2h) on the two
  trimming shells.
- `plugin_ames` and `compute_adaptive_h` are now public.

### Unchanged

- The adaptive step size, including the floor of `0.5` for integer-valued
  features. It was removed and restored between releases; users upgrading from
  0.1.x see no change.
- `MarginfxResult.estimates`, `.std_errors` and `.conf_int`.

---

## 0.1.2 — 2026-07-08

Initial public releases. Plug-in average marginal effects with refitting
bootstrap standard errors.
