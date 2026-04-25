"""
engines/pytorch.py
------------------
PyTorch engine for marginfx.

Key differences from the TensorFlow engine:
    - Uses torch.autograd for exact gradients instead of tf.GradientTape
    - No .fit() method — bootstrap refit loop is written explicitly
    - Model modes matter: model.eval() for inference, model.train() for refit
    - Optimizer and loss function must be provided by user (PyTorch is explicit)

Warm-start:
    - copy.deepcopy() preserves all model weights cleanly
    - Continue training on bootstrap sample for n_epochs (default 10)
    - Fresh optimizer instance per replicate via optimizer_fn callable

Gradient computation:
    - requires_grad=True on input tensor
    - torch.autograd.grad() gives exact gradients for all features at once
    - Falls back to first differences for categorical features

Common loss presets are provided as module-level constants for convenience.
"""

import numpy as np
import copy
from typing import Callable, Tuple, Optional


# ---------------------------------------------------------------------------
# Common loss function presets
# ---------------------------------------------------------------------------

def _get_default_losses():
    """Lazy import of torch loss functions."""
    import torch.nn as nn
    return {
        'binary': nn.BCELoss(),
        'multiclass': nn.CrossEntropyLoss(),
        'regression': nn.MSELoss(),
    }


# Default optimizer: Adam with lr=1e-3
DEFAULT_OPTIMIZER_FN = lambda params: __import__('torch').optim.Adam(
    params, lr=1e-3
)


# ---------------------------------------------------------------------------
# Predict function
# ---------------------------------------------------------------------------

def make_predict_fn(model) -> Callable:
    """
    Build a predict_fn for a fitted PyTorch model.

    Autodetects output type from output shape:
        - Single output unit or sigmoid: returns raw output (regression or
          binary classification probability)
        - Two output units + softmax: returns P(y=1)
        - Multiple output units: returns full output matrix

    Always runs in model.eval() mode with torch.no_grad().

    Parameters
    ----------
    model : torch.nn.Module
        A fitted PyTorch model.

    Returns
    -------
    Callable
        predict_fn(X) -> np.ndarray of shape (n_obs,)
    """
    import torch

    def predict_fn(X: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            output = model(X_tensor)

            # Convert to numpy
            output_np = output.numpy()

            # Squeeze (n, 1) -> (n,) for regression / binary classification
            if output_np.ndim == 2 and output_np.shape[1] == 1:
                return output_np.squeeze(axis=1)

            # Two output units — return P(y=1)
            if output_np.ndim == 2 and output_np.shape[1] == 2:
                return output_np[:, 1]

            return output_np

    return predict_fn


# ---------------------------------------------------------------------------
# Exact gradient computation via torch.autograd
# ---------------------------------------------------------------------------

def make_gradient_ame_fn(model) -> Callable:
    """
    Build an AME function using exact gradients from torch.autograd.

    Replaces finite differences in core.py with exact gradients.
    One autograd pass gives gradients for all features simultaneously.

    For categorical features, falls back to first differences since
    gradients with respect to discrete inputs are not meaningful.

    Parameters
    ----------
    model : torch.nn.Module
        A fitted PyTorch model.

    Returns
    -------
    Callable
        gradient_ame_fn(X, feature_idx, is_categorical, h) -> np.ndarray
        Returns pointwise marginal effects for a single feature, shape (n_obs,)
    """
    import torch

    def gradient_ame_fn(
        X: np.ndarray,
        feature_idx: int,
        is_categorical: bool = False,
        h: float = 1e-4,
    ) -> np.ndarray:

        model.eval()

        # Categorical: first difference, no gradient needed
        if is_categorical:
            with torch.no_grad():
                X_0 = X.copy()
                X_1 = X.copy()
                X_0[:, feature_idx] = 0.0
                X_1[:, feature_idx] = 1.0

                pred_0 = model(
                    torch.tensor(X_0, dtype=torch.float32)
                ).numpy()
                pred_1 = model(
                    torch.tensor(X_1, dtype=torch.float32)
                ).numpy()

                pred_0 = _squeeze_output(pred_0)
                pred_1 = _squeeze_output(pred_1)

                return pred_1 - pred_0

        # Continuous: exact gradient via autograd
        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)

        output = model(X_tensor)
        output = _squeeze_output_tensor(output)

        # Sum over observations to get scalar for backward pass
        # Gradient of sum(output) w.r.t. X gives dy_i/dx_i per row
        grad_outputs = torch.ones_like(output)
        grads = torch.autograd.grad(
            outputs=output,
            inputs=X_tensor,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False,
        )[0]

        # Return gradient for requested feature only, shape (n_obs,)
        return grads[:, feature_idx].detach().numpy()

    return gradient_ame_fn


def _squeeze_output(output_np: np.ndarray) -> np.ndarray:
    """Squeeze model output to 1D numpy array."""
    if output_np.ndim == 2 and output_np.shape[1] == 1:
        return output_np.squeeze(axis=1)
    if output_np.ndim == 2 and output_np.shape[1] == 2:
        return output_np[:, 1]
    return output_np


def _squeeze_output_tensor(output):
    """Squeeze tensor output for autograd compatibility."""
    import torch
    if len(output.shape) == 2 and output.shape[1] == 1:
        return output.squeeze(dim=1)
    if len(output.shape) == 2 and output.shape[1] == 2:
        return output[:, 1]
    return output


# ---------------------------------------------------------------------------
# Warm-start fit function
# ---------------------------------------------------------------------------

def make_fit_fn(
    model,
    optimizer_fn: Callable = DEFAULT_OPTIMIZER_FN,
    loss_fn=None,
    n_epochs: int = 10,
    batch_size: int = 32,
) -> Callable:
    """
    Build a fit_fn for a PyTorch model.

    Warm-starts by copying model weights and continuing training on the
    bootstrap sample for n_epochs. Each replicate gets a fresh optimizer
    instance via optimizer_fn to avoid state contamination across replicates.

    Parameters
    ----------
    model : torch.nn.Module
        Original fitted PyTorch model.
    optimizer_fn : Callable
        Callable that takes model parameters and returns an optimizer.
        Default: lambda params: torch.optim.Adam(params, lr=1e-3)
        Example: lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9)
    loss_fn : torch loss function, optional
        Loss function for refit. Default: BCELoss for binary classification.
        Common options:
            torch.nn.BCELoss()          -- binary classification
            torch.nn.MSELoss()          -- regression
            torch.nn.CrossEntropyLoss() -- multiclass classification
    n_epochs : int
        Number of training epochs per bootstrap replicate. Default 10.
    batch_size : int
        Mini-batch size for refit loop. Default 32.

    Returns
    -------
    Callable
        fit_fn(model, X_boot, y_boot) -> fitted_model
    """
    import torch

    if loss_fn is None:
        loss_fn = torch.nn.BCELoss()

    def fit_fn(current_model, X_boot: np.ndarray, y_boot: np.ndarray):
        # Deep copy preserves all weights as warm-start initialization
        new_model = copy.deepcopy(current_model)
        new_model.train()

        # Fresh optimizer instance for this replicate
        optimizer = optimizer_fn(new_model.parameters())

        X_tensor = torch.tensor(X_boot, dtype=torch.float32)
        y_tensor = torch.tensor(y_boot, dtype=torch.float32)

        n = X_tensor.shape[0]
        n_batches = max(1, n // batch_size)

        for epoch in range(n_epochs):
            # Shuffle each epoch
            perm = torch.randperm(n)
            X_shuffled = X_tensor[perm]
            y_shuffled = y_tensor[perm]

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                optimizer.zero_grad()
                output = new_model(X_batch)
                output = _squeeze_output_tensor(output)

                loss = loss_fn(output, y_batch)
                loss.backward()
                optimizer.step()

        new_model.eval()
        return new_model

    return fit_fn


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_engine(
    model,
    optimizer_fn: Callable = DEFAULT_OPTIMIZER_FN,
    loss_fn=None,
    n_epochs: int = 10,
    batch_size: int = 32,
) -> Tuple[Callable, Callable, Callable]:
    """
    Get predict_fn, fit_fn, and gradient_ame_fn for a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        Fitted PyTorch model.
    optimizer_fn : Callable
        Returns a fresh optimizer given model parameters.
        Default: Adam with lr=1e-3.
        Example: lambda params: torch.optim.SGD(params, lr=0.01)
    loss_fn : torch loss function, optional
        Default: BCELoss (binary classification).
        Override for regression: torch.nn.MSELoss()
    n_epochs : int
        Bootstrap refit epochs. Default 10.
    batch_size : int
        Bootstrap refit batch size. Default 32.

    Returns
    -------
    Tuple[Callable, Callable, Callable]
        (predict_fn, fit_fn, gradient_ame_fn)

        predict_fn(X) -> np.ndarray
            Standard predictions in eval mode.

        fit_fn(model, X_boot, y_boot) -> fitted_model
            Warm-start refit on bootstrap sample.

        gradient_ame_fn(X, feature_idx, is_categorical, h) -> np.ndarray
            Exact pointwise marginal effects via torch.autograd.

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> from engines.pytorch import get_engine
    >>> from bootstrap import bootstrap_ames
    >>>
    >>> model = MyNet()  # your fitted torch.nn.Module
    >>> predict_fn, fit_fn, gradient_ame_fn = get_engine(
    ...     model,
    ...     optimizer_fn=lambda p: torch.optim.Adam(p, lr=1e-3),
    ...     loss_fn=nn.BCELoss(),
    ...     n_epochs=10,
    ... )
    >>> result = bootstrap_ames(
    ...     model, X, y, fit_fn, predict_fn,
    ...     gradient_ame_fn=gradient_ame_fn
    ... )
    >>> result.summary()
    """
    predict_fn = make_predict_fn(model)
    fit_fn = make_fit_fn(
        model,
        optimizer_fn=optimizer_fn,
        loss_fn=loss_fn,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )
    gradient_ame_fn = make_gradient_ame_fn(model)

    return predict_fn, fit_fn, gradient_ame_fn
