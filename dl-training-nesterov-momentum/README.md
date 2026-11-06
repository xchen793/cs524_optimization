# Nesterov Momentum

Intermediate | deep-learning | optimization | training

### The problem, from first principles

Plain momentum can overshoot in narrow valleys because its velocity keeps pushing the parameters forward even after the gradient has turned around.

This is the problem addressed by [SGD with momentum](/ide/dl-training-sgd-momentum). Nesterov momentum lets the gradient look one step ahead before committing to the parameter update.

In this exercise, use this reparameterized form of Nesterov's method:

$$
v_{\text{new}} = \text{momentum} \cdot v + \text{grad}
$$

$$
\text{param}_{\text{new}}
=
\text{param}
-
\text{lr}
\left(
\text{grad}
+
\text{momentum} \cdot v_{\text{new}}
\right)
$$

The gradient is evaluated at the current parameter. Initial velocity is zeros.

This is a reparameterized form of Nesterov's method. It matches the textbook update up to a change of variables, which is how PyTorch's `SGD(nesterov=True)` is written.

### From theory to code

Implement:

```python
nesterov_step(param, grad, velocity, lr, momentum)
```

The function returns:

```python
(new_param, new_velocity)
```

The signature is provided in the editor.

### Constraints

- `param`, `grad`, and `velocity` must have the same shape. A mismatch raises `ValueError`.
- `momentum` must be in `[0, 1)`. Otherwise, raise `ValueError`.
- Inputs must not be modified.
- Return new arrays.
- With `momentum = 0`, the result must equal plain SGD.

### Hints

<details>
<summary>Hint 1</summary>

Compute the new velocity first.

</details>

<details>
<summary>Hint 2</summary>

The parameter update uses the new velocity, not the old velocity.

</details>

<details>
<summary>Hint 3</summary>

Write both update lines and compare them with the momentum version.

</details>
