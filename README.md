# marginfx

**Average marginal effects and bootstrap standard errors for any machine learning model.**

Get OLS-style interpretability from scikit-learn, XGBoost, TensorFlow, and PyTorch. One function call. One tidy table.

```python
import marginfx as mfx

model = RandomForestClassifier().fit(X_train, y_train)
result = mfx.fit(model, X, y, feature_names=feature_names)
result.summary()
```

```
=================================================================
marginfx: Average Marginal Effects
=================================================================
Observations: 1000
Bootstrap replicates: 200
Confidence level: 95%
-----------------------------------------------------------------
        term  estimate  std_error  statistic   p_value  conf_low  conf_high
         age     0.032      0.004      8.100     0.000     0.024      0.040
      income     0.008      0.001      6.300     0.000     0.006      0.010
      female    -0.012      0.003     -3.900     0.000    -0.018     -0.006
   education     0.021      0.005      4.200     0.000     0.011      0.031
=================================================================
```

---

## What is this?

In classical econometrics, OLS gives you a coefficient table — estimates, standard errors, p-values — in units that are immediately interpretable. A one-unit increase in age increases income by $X. Everyone understands that.

Modern ML models (random forests, neural nets, gradient boosting) give you better predictions but no such table. You get a black box.

**marginfx bridges the gap.** It computes *average marginal effects* (AMEs) — the same quantity that OLS reports as its coefficients — for any model. A one-unit increase in age increases P(default) by 0.032 percentage points, regardless of whether the underlying model is a random forest or a neural net.

Standard errors come from a nonparametric bootstrap with warm-start reinitialization, making the computation practical even for expensive models. For TensorFlow and PyTorch, exact gradients replace finite differences automatically.

The output is a tidy DataFrame, directly inspired by the [`broom`](https://broom.tidymodels.org/) package in R and the [`marginaleffects`](https://marginaleffects.com/) package — now available for the Python ML ecosystem.

---

## Installation

```bash
pip install marginfx
```

Install with the ML frameworks you use:

```bash
pip install marginfx[sklearn]              # scikit-learn + XGBoost + LightGBM
pip install marginfx[tensorflow]           # TensorFlow / Keras
pip install marginfx[pytorch]              # PyTorch
pip install marginfx[all]                  # everything
```

---

## Quick start

### scikit-learn

```python
import marginfx as mfx
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)

result = mfx.fit(
    model, X, y,
    feature_names=feature_names,
    n_bootstrap=200,
    seed=42,
)

result.summary()          # formatted table
result.tidy()             # pandas DataFrame
```

### XGBoost

```python
import marginfx as mfx
import xgboost as xgb

model = xgb.XGBClassifier().fit(X_train, y_train)
result = mfx.fit(model, X, y, feature_names=feature_names, seed=42)
result.summary()
```

### TensorFlow / Keras

```python
import marginfx as mfx
import tensorflow as tf

model = tf.keras.models.load_model("my_model.keras")

result = mfx.fit(
    model, X, y,
    feature_names=feature_names,
    n_epochs=10,       # bootstrap warm-start epochs
    seed=42,
)
result.summary()
```

### PyTorch

```python
import marginfx as mfx
import torch.nn as nn

result = mfx.fit(
    model, X, y,
    feature_names=feature_names,
    loss_fn=nn.BCELoss(),
    optimizer_fn=lambda p: torch.optim.Adam(p, lr=1e-3),
    n_epochs=10,
    seed=42,
)
result.summary()
```

### Pandas DataFrames

```python
# Column names are picked up automatically
result = mfx.fit(model, df[features], df["target"])
result.tidy()
```

### Categorical features

```python
# Categorical features use first differences (0 -> 1) instead of derivatives
result = mfx.fit(
    model, X, y,
    feature_names=feature_names,
    categorical_features=["female", "married", "has_degree"],
)
```

---

## The tidy output

`result.tidy()` returns a pandas DataFrame modeled on `broom::tidy()` in R:

| term | estimate | std_error | statistic | p_value | conf_low | conf_high |
|------|----------|-----------|-----------|---------|----------|-----------|
| age | 0.032 | 0.004 | 8.10 | 0.000 | 0.024 | 0.040 |
| income | 0.008 | 0.001 | 6.30 | 0.000 | 0.006 | 0.010 |
| female | -0.012 | 0.003 | -3.90 | 0.000 | -0.018 | -0.006 |

- **estimate** — the average marginal effect (AME): mean of pointwise dy/dx across all observations
- **std_error** — bootstrap standard deviation across replicates
- **statistic** — estimate / std_error (normal approximation)
- **p_value** — two-tailed p-value under normal approximation
- **conf_low / conf_high** — percentile bootstrap confidence interval

---

## How it works

### Average marginal effects

For a continuous feature *x_j*, the marginal effect at observation *i* is:

```
ME_i(x_j) = ∂f(x_i) / ∂x_j
```

Approximated via central finite differences:

```
ME_i(x_j) ≈ [f(x_i + h·e_j) - f(x_i - h·e_j)] / 2h
```

The AME is the mean across all observations:

```
AME(x_j) = (1/n) Σ ME_i(x_j)
```

For binary/categorical features, a first difference replaces the derivative:

```
ME_i(x_j) = f(x_i | x_j=1) - f(x_i | x_j=0)
```

For TensorFlow and PyTorch models, `tf.GradientTape` and `torch.autograd` provide exact gradients, replacing finite differences automatically.

### Bootstrap standard errors

Standard errors come from a nonparametric bootstrap:

1. Resample the data with replacement
2. Refit the model warm-starting from the original (faster convergence)
3. Compute AMEs on the bootstrap sample
4. Repeat B times
5. SE = standard deviation of the B AME estimates
6. CI = percentile interval of the B AME estimates

Warm-starting from the original model makes the bootstrap practical for expensive models — bootstrap replicates converge in far fewer iterations than cold retraining.

---

## Supported models

| Framework | Models | Gradient method | Warm-start |
|-----------|--------|-----------------|------------|
| scikit-learn | RandomForest, GradientBoosting, LogisticRegression, LinearRegression, SVC, and all sklearn-compatible models | Finite differences | Yes (where supported) |
| XGBoost | XGBClassifier, XGBRegressor | Finite differences | Yes (native) |
| LightGBM | LGBMClassifier, LGBMRegressor | Finite differences | Yes (native) |
| TensorFlow | tf.keras.Model | Exact (GradientTape) | Yes (continued training) |
| PyTorch | torch.nn.Module | Exact (autograd) | Yes (continued training) |

Model type is detected automatically. No need to specify the engine.

---

## API reference

### `mfx.fit()`

```python
mfx.fit(
    model,                        # fitted model — any supported type
    X,                            # feature matrix (numpy array or pandas DataFrame)
    y,                            # target vector
    feature_names=None,           # list of feature names (auto from DataFrame columns)
    categorical_features=None,    # list of categorical feature indices or names
    n_bootstrap=200,              # number of bootstrap replicates
    alpha=0.05,                   # significance level (0.05 = 95% CI)
    seed=None,                    # random seed for reproducibility
    verbose=True,                 # print bootstrap progress
    h=1e-4,                       # finite difference step size (sklearn models)
    n_epochs=10,                  # bootstrap refit epochs (TF/PyTorch)
    batch_size=32,                # bootstrap refit batch size (TF/PyTorch)
    optimizer_fn=None,            # optimizer callable (PyTorch only)
    loss_fn=None,                 # loss function (PyTorch only)
)
```

Returns a `MarginfxResult` object.

### `MarginfxResult`

```python
result.tidy()        # pandas DataFrame with estimates, SEs, CIs
result.summary()     # formatted summary table printed to stdout
result.estimates     # dict of feature -> AME estimate
result.std_errors    # dict of feature -> bootstrap SE
result.conf_int      # dict of feature -> (conf_low, conf_high)
result.n_obs         # number of observations
result.n_bootstrap   # number of bootstrap replicates
```

---

## Citation

If you use marginfx in published research, please cite:

```bibtex
@inproceedings{marginfx2026,
  title     = {marginfx: Average Marginal Effects for Any Machine Learning Model},
  author    = {Your Name},
  booktitle = {Proceedings of the 26th IEEE International Conference on Data Mining (ICDM)},
  year      = {2026},
  address   = {Shenyang, China},
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
