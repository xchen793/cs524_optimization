import numpy as np


def mse_loss_forward(input: np.ndarray, target: np.ndarray, reduction: str = "mean"):
    """
    Mirrors torch.nn.functional.mse_loss(input, target, reduction=reduction).

    reduction:
        "mean": average squared error over every element (the default).
        "sum":  total squared error, no averaging.
        "none": the per-element squared error itself, no reduction at all.

    Variable	        Shape	Example
    input	            (N, D)	(4, 3)
    target	            (N, D)	(4, 3)
    input - target	    (N, D)	(4, 3), elementwise difference
    squared_error	    (N, D)	(4, 3), elementwise square
    return for "mean"	(), a scalar	one number: sum of all 12 values / 12
    return for "sum"	(), a scalar	one number: sum of all 12 values
    return for "none"	(N, D)	(4, 3), same as squared_error
    """
    squared_error = (input - target) ** 2
    if reduction == "mean":
        return squared_error.mean()
    if reduction == "sum":
        return squared_error.sum()
    return squared_error


def mse_loss_backward(
    input: np.ndarray, target: np.ndarray, reduction: str = "mean", grad_output=1.0
) -> np.ndarray:
    """
    dL/d_input for MSE. The unreduced per-element gradient is always
    2*(input - target); "mean" divides that by the element count (since
    the forward pass divided the sum by that same count), "sum" and
    "none" leave it as is.

    grad_output: the upstream gradient flowing in from whatever used
    this loss's output (almost always 1.0, since a loss is usually the
    very end of the graph, but kept general to chain correctly either way).
    """
    grad = 2.0 * (input - target)
    if reduction == "mean":
        grad = grad / input.size
    return grad_output * grad
