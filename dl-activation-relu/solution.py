import numpy as np

'''
x           = np.array([-2.0, 0.0, 3.0], dtype=np.float32)
grad_output = np.array([ 5.0, 5.0, 5.0], dtype=np.float32)

mask = (x > 0)                          # [False, False, True], dtype=bool
mask = mask.astype(grad_output.dtype)   # [0., 0., 1.], dtype=float32
grad_output * mask                      # [0., 0., 5.], dtype=float32
'''

def relu_forward(x: np.ndarray) -> np.ndarray:
    """Mirrors torch.relu(x) / F.relu(x): max(0, x), elementwise."""
    return x * (x > 0) # np.maximum(0.0, x)


def relu_backward(grad_output: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    grad_output: the gradient flowing back from whatever came after this
    ReLU (dL/d(relu_output)), same shape as x.
    x: the ORIGINAL input this ReLU was applied to during the forward pass.

    Returns:
        dL/dx = grad_output * 1 where x > 0, grad_output * 0 where x <= 0
        (PyTorch's convention: the gradient at exactly x == 0 is 0).
    """
    return grad_output * (x > 0).astype(grad_output.dtype)
