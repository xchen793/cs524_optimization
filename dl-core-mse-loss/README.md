# MSE

Beginner | deep-learning | loss-functions | autograd

### The problem, from first principles

Every activation earlier in this section has a backward pass that flows into _something_ — eventually, that something is a loss function, the single number a whole network is trained to minimize. Mean squared error is the default choice for regression: it turns a whole batch of (prediction, target) pairs into one scalar by penalizing the squared distance between each pair.

### From theory to code

Implement `mse_loss_forward(input, target, reduction="mean")` and `mse_loss_backward(input, target, reduction="mean", grad_output=1.0)`, mirroring `torch.nn.functional.mse_loss`'s three reduction modes: `"mean"` (default), `"sum"`, `"none"`.

### Constraints

- `input`, `target`: matching NumPy array shapes.
- `reduction="mean"`: returns a scalar, the average squared error.
- `reduction="sum"`: returns a scalar, the total squared error.
- `reduction="none"`: returns the full elementwise squared-error array, same shape as `input`.
- `mse_loss_backward` takes the same `input`/`target`/`reduction`, plus `grad_output` (the upstream gradient, `1.0` by default), and returns a same-shape gradient array regardless of `reduction`.
- No Python loops; no mutating `input` or `target`.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

The elementwise squared error is the same regardless of `reduction` — only the final reduction step (`.mean()`, `.sum()`, or nothing) differs.

</details>

<details>
<summary>Hint 2</summary>

Whatever division the forward pass applies for `"mean"` (dividing by `input.size`), the backward pass must apply that exact same division to its gradient — otherwise gradients would silently scale with batch size.

</details>
