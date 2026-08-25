# marginfx

**Window average marginal effects, with valid inference, for any machine learning model.**

Get an OLS-style coefficient table out of scikit-learn, XGBoost, TensorFlow, or PyTorch — estimate, standard error, *t*-statistic, p-value — with standard errors that are asymptotically valid rather than heuristic.

```python
import marginfx as mfx
from sklearn.ensemble import RandomForestClassifier

result = mfx.fit(RandomForestClassifier(), X, y, feature_names=feature_names)
result.summary()
```

```
========================================================================================
marginfx: Window Average Marginal Effects
========================================================================================
Observations: 1000
Estimator: debiased cross-fitted (K=5)
Standard errors: influence function
Confidence level: 95%
----------------------------------------------------------------------------------------
        term  estimate         h  std_error  statistic   p_value  conf_low  conf_high  trimmed
         age     0.032  0.685000      0.004      8.100     0.000     0.024      0.040    0.004
      income     0.008  0.372000      0.001      6.300     0.000     0.006      0.010    0.003
      female    -0.012       NaN      0.003     -3.900     0.000    -0.018     -0.006    0.000
   education     0.021  0.115000      0.005      4.200     0.000     0.011      0.031    0.006
========================================================================================
```

> **Note:** `mfx.fit` takes an **unfitted** learner. It fits the model itself, once per cross-fitting fold. This is not a convenience — cross-fitting requires that the model never see the observations at which its own score is evaluated.

---

## What is this?

In classical econometrics, OLS gives you a coefficient table in units that are immediately interpretable. Modern ML models predict better but give you no such table.

**marginfx bridges the gap**, and — unlike SHAP or permutation importance — it comes with an asymptotic theory saying what population quantity is being estimated and how uncertain the estimate is.

---

## The estimand

marginfx targets the **window average marginal effect** at scale `h`:

```
θ_{j,h} = E[ w(X) · ( f(X + h·e_j) − f(X − h·e_j) ) / 2h ]
```

Two things distinguish this from "a finite-difference approximation to a derivative":

**`h` is part of the target, not a numerical tolerance.** The centered difference *is* the defining operation, computed exactly. `h` is chosen and reported the way a bandwidth is; the default asks how the prediction responds to a displacement of a twentieth of a standard deviation. It appears in the output table.

**No derivative need exist anywhere.** `θ_{j,h}` is well defined for any bounded prediction function, which is what lets a single estimand cover neural networks and regression trees at once. It converges to the classical AME `E[∂f/∂x_j]` as `h → 0` whenever `f` has one weak derivative.

The **trimming weight** `w` gives weight zero to observations within `h_j` of the boundary of the observed support, where `x ± h·e_j` would fall outside the data and the fitted model would be silently extrapolating. The trimmed shell carries probability mass of order `h_j`; the fraction excluded is reported in the `trimmed` column.

Default step size:

```
h_j = max(1e-4, 0.05 · σ̂_j)
```

For binary features the difference is replaced by the contrast `f(x | x_j=1) − f(x | x_j=0)`.

---

## How inference works

`mfx.fit` returns a **debiased, cross-fitted** estimator. Split the sample into `K` folds; for each fold, fit the learner and a Riesz representer on the other `K−1` folds, then evaluate on the held-out fold:

```
θ̂ = (1/n) Σ_k Σ_{i∈I_k} [ w(xᵢ)·D_h f̂^(−k)(xᵢ)  +  α̂^(−k)(xᵢ)·(yᵢ − f̂^(−k)(xᵢ)) ]
```

The first term is the plug-in average. The second removes, at first order, the bias that regularization of the learner transmits to that average — each held-out residual, weighted by the representer, testifies to how the learner is locally mis-calibrated where that mis-calibration matters for the marginal effect.

The representer is estimated by **Riesz regression**, minimizing

```
L(α) = (1/n) Σᵢ [ α(xᵢ)² − 2·wᵢ·( α(xᵢ + h·e_j) − α(xᵢ − h·e_j) ) / 2h ]
```

which needs only the ability to evaluate candidates at shifted points. **No density is ever estimated.** For the default polynomial sieve this loss is quadratic, so it is solved in closed form.

**Standard errors require no resampling.** The summands, centered at `θ̂`, are the influence function values; the standard error is their standard deviation over `√n`. The whole table costs `K` model fits instead of the hundreds a bootstrap needs.

For simultaneous bands across features, pass `n_multiplier=500` to run a multiplier bootstrap over the influence values — no additional model fits.

### Double robustness

The moment bias is exactly the *product* of the two nuisance errors, so an error in the learner harms the estimate only to the extent the representer is also wrong. If the representer is known in closed form — as in a simulation design with known covariate density — the bias vanishes identically and valid inference requires no consistency from the learner at all. Its failures are paid in variance, never in location.

```python
from marginfx import gaussian_window_riesz

result = mfx.fit(
    learner, X, y,
    trim=False,                                    # Gaussian support is unbounded
    riesz=lambda idx, h, cat: gaussian_window_riesz(idx, h),
)
```

---

## Installation

```bash
pip install marginfx
```

```bash
pip install marginfx[sklearn]      # scikit-learn + XGBoost + LightGBM
pip install marginfx[tensorflow]   # TensorFlow / Keras
pip install marginfx[pytorch]      # PyTorch
pip install marginfx[all]          # everything
```

Core requirements are numpy, pandas and scipy only.

---

## Quick start

### scikit-learn / XGBoost / LightGBM

Pass the estimator unfitted; it is cloned and refit per fold.

```python
import marginfx as mfx
from sklearn.ensemble import RandomForestClassifier

result = mfx.fit(
    RandomForestClassifier(n_estimators=200),
    X, y,
    feature_names=feature_names,
    seed=42,
)
result.summary()
result.tidy()
```

### TensorFlow / Keras

Pass a **factory** — a zero-argument callable returning a fresh compiled model.

```python
import marginfx as mfx
import tensorflow as tf

def make_model():
    m = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy")
    return m

result = mfx.fit(make_model, X, y, n_epochs=20, seed=42)
```

### PyTorch

```python
import marginfx as mfx
import torch, torch.nn as nn

result = mfx.fit(
    lambda: nn.Sequential(nn.Linear(p, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()),
    X, y,
    loss_fn=nn.BCELoss(),
    optimizer_fn=lambda prm: torch.optim.Adam(prm, lr=1e-3),
    n_epochs=20,
    seed=42,
)
```

### Categorical features

Binary features use the contrast with an inverse propensity representer.

```python
result = mfx.fit(
    model, X, y,
    categorical_features=["female", "married", "has_degree"],
)
```

### Pandas

```python
result = mfx.fit(model, df[features], df["target"])   # column names picked up
```

---

## The tidy output

| column | meaning |
|--------|---------|
| `term` | feature name |
| `estimate` | window AME `θ̂_{j,h}` |
| `h` | step size defining the estimand (`NaN` for categoricals) |
| `std_error` | influence-function standard error |
| `statistic` | `estimate / std_error` |
| `p_value` | two-tailed, asymptotic normal |
| `conf_low` / `conf_high` | pointwise confidence interval |
| `simul_low` / `simul_high` | simultaneous band (with `n_multiplier > 0`) |
| `trimmed` | fraction of observations given trimming weight zero |

---

## The bootstrap is a diagnostic, not inference

`mfx.bootstrap_diagnostic` retains the older refitting bootstrap. It measures how sensitive the reported effects are to resampling, refitting and hyperparameter selection — large dispersion is a useful warning that the fitted model is unstable.

**It is not a valid standard error**, for two reasons:

1. It recenters at the fitted model, so it is blind to the regularization bias the correction term removes. Every replicate recenters at a similarly regularized fit. In the paper's simulations this drives random-forest coverage down to roughly 0.10 at n = 5,000 — the interval ends up about as wide as the bias itself. This is not specific to forests, but to any learner whose L² bias shrinks more slowly than its intervals.
2. Because each replicate is evaluated at the original sample, it omits the variance from averaging a heterogeneous effect over a finite sample of covariate values.

```python
model = RandomForestClassifier().fit(X, y)          # note: already fitted
diag = mfx.bootstrap_diagnostic(model, X, y, n_bootstrap=200)
```

---

## API reference

### `mfx.fit()`

```python
mfx.fit(
    learner,                      # UNFITTED estimator, or zero-arg factory callable
    X, y,
    feature_names=None,
    categorical_features=None,
    h='adaptive',                 # 'adaptive', a float, or a per-feature array
    trim=True,                    # apply the trimming weight
    n_folds=5,                    # cross-fitting folds K
    riesz=None,                   # None estimates it; or dict / factory for known α
    sieve_degree=2,               # polynomial degree for the default sieve
    sieve_ridge=1e-6,
    alpha_bound=None,             # truncation bound for the representer
    alpha=0.05,
    n_multiplier=0,               # multiplier bootstrap draws for simultaneous bands
    seed=None,
    verbose=True,
    n_epochs=10, batch_size=32,   # Keras / PyTorch
    optimizer_fn=None, loss_fn=None,
)
```

### `MarginfxResult`

```python
result.tidy()                     # pandas DataFrame
result.summary()                  # formatted table
result.estimates                  # feature -> θ̂
result.std_errors                 # feature -> SE
result.conf_int                   # feature -> (low, high)
result.simultaneous_conf_int      # feature -> (low, high), if n_multiplier > 0
result.h                          # feature -> step size used
result.trimmed_fraction           # feature -> fraction trimmed
result.influence                  # feature -> per-observation influence values
result.method                     # 'debiased' | 'bootstrap-diagnostic'
result.n_folds, result.n_obs
```

---

## Citation

```bibtex
@article{marginfx2026,
  title  = {Model-Agnostic Average Marginal Effects with Valid Inference},
  author = {Parker, Jason},
  year   = {2026},
}
```

---

## Related work

- [`marginaleffects`](https://marginaleffects.com/) — the R package that inspired this project
- [`broom`](https://broom.tidymodels.org/) — tidy model output in R
- [`shap`](https://github.com/shap/shap) — SHAP values for model explanation
- [`lime`](https://github.com/marcotcr/lime) — local interpretable model-agnostic explanations

---

## License

MIT
