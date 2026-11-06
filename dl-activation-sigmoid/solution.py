import numpy as np


def sigmoid_forward(x: np.ndarray) -> np.ndarray:
    """Mirrors torch.sigmoid(x): 1 / (1 + exp(-x)). Clip x before exponentiating for numerical stability."""
    clipped = np.clip(x, -500, 500) # 截断不会影响精度。sigmoid(500) 在浮点数里本来就等于 1.0，sigmoid(-500) 约等于 7e-218，和真实值几乎没有区别。

    # 500 这个界限只对 float64 安全。float32 最大约 3.4e38，exp(89) 就溢出了：
    '''
    def sigmoid_forward(x):
        out = np.empty_like(x, dtype=float)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))   # x>=0：exp(-x) ≤ 1，不会溢出
        ex = np.exp(x[~pos])                         # x<0：exp(x) < 1，不会溢出
        out[~pos] = ex / (1.0 + ex)                  # 与 1/(1+e^{-x}) 在数学上相等
        return out
    '''
    return 1.0 / (1.0 + np.exp(-clipped))


def sigmoid_backward(grad_output: np.ndarray, output: np.ndarray) -> np.ndarray:
    """
    grad_output: dL/d(sigmoid_output).
    output: the FORWARD PASS'S OWN OUTPUT (sigmoid(x)), not x itself.

    Returns:
        dL/dx = grad_output * output * (1 - output).

    Sigmoid's derivative, f'(x) = f(x)*(1-f(x)), is written entirely in
    terms of f(x) itself -- the forward pass's saved output is enough,
    there's no need to keep the original x around at all, or recompute
    sigmoid a second time.
    """
    return grad_output * output * (1 - output)
