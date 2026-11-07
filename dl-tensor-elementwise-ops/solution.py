import numpy as np


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mirrors torch.add(a, b) / a + b."""
    return a + b


def sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mirrors torch.sub(a, b) / a - b."""
    return a - b


def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mirrors torch.mul(a, b) / a * b."""
    return a * b


def div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Mirrors torch.div(a, b): TRUE division, always -- even when both a
    and b are integer tensors, the result is floating-point, never
    truncated toward zero the way `//` would.
    """
    # TODO: np.true_divide, not a plain `/` on integer arrays and not `//`.
    # np.floor_divide(a, b)      # identical to a // b
    return np.true_divide(a,b)


def power(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mirrors torch.pow(a, b) / a ** b."""
    return a**b
