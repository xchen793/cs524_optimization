import numpy as np


def tanh_forward(x: np.ndarray) -> np.ndarray:
    """Mirrors torch.tanh(x)."""
    # tanh(x) = sign(x) * (1 - e^{-2|x|}) / (1 + e^{-2|x|})
    e = np.exp(-2.0 * np.abs(x))   # always in (0, 1], so it can never overflow
    return np.sign(x) * (1.0 - e) / (1.0 + e)

    
def tanh_backward(grad_output: np.ndarray, output: np.ndarray) -> np.ndarray:
    """
    output: the forward pass's own saved result (tanh(x)), the same
    "backward needs the output, not x" pattern 02-sigmoid uses.

    Returns:
        dL/dx = grad_output * (1 - output^2).
    """
    return grad_output * (1 - output * output)
    pass
