import numpy as np


def make_tensor(data, dtype: np.dtype | None = None) -> np.ndarray:
    """
    Mirrors torch.tensor(data, dtype=dtype)'s dtype-inference rule,
    which differs from NumPy's own default:
      - if dtype is given, cast to it, no inference needed.
      - otherwise, if the inferred dtype is floating-point, use
        float32 (PyTorch's real default float dtype), NOT NumPy's own
        default of float64.
      - otherwise (integers, bools), keep whatever NumPy inferred.
    """
    # TODO: np.array(data) first, then apply the rule above.
    array = np.array(data) 
    if dtype is not None: 
        return array.astype(dtype) # 返回一个新数组
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float32) 
    return array


def zeros(shape: tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
    """Mirrors torch.zeros(shape, dtype=dtype) -- PyTorch's own default dtype is float32."""

    # Thin wrapper over np.zeros that mirrors torch.zeros.
    # Only difference: default dtype is float32 (NumPy defaults to float64),
    # so arrays stay consistent with PyTorch and float64 doesn't leak into later ops.
    return np.zeros(shape, dtype=dtype)
    


def ones(shape: tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
    """Mirrors torch.ones(shape, dtype=dtype)."""
    # Thin wrapper over np.ones that mirrors torch.ones; same idea as zeros().
    # Default dtype is float32 to match PyTorch (NumPy would give float64),
    # so we avoid silent float64 upcasting in downstream ops.
    return np.ones(shape, dtype=dtype)


def arange(start: float, stop: float, step: float = 1, dtype: np.dtype | None = None) -> np.ndarray:
    """Mirrors torch.arange(start, stop, step, dtype=dtype), same float32-default rule as make_tensor."""
    # TODO: np.arange(start, stop, step), then the same dtype rule
    # make_tensor uses.
    result = np.arange(start, stop, step)
    if dtype is not None:
        return result.astype(dtype)
    if np.issubdtype(result.dtype, np.floating):
        return result.astype(np.float32)
    return result
