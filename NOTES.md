# Notes and derivations

## Stage 1: micrograd

### Reverse-mode differentiation
Why does one backward pass give the gradient with respect to every input?
Why must nodes be processed in reverse topological order?

### Local derivatives
For each operation, the gradient it passes to its inputs given the upstream gradient $\bar{o} = \partial L / \partial o$:

- Addition, $o = a + b$:
- Multiplication, $o = a b$:
- Power, $o = a^k$:
- Exponential, $o = e^a$:
- Tanh, $o = \tanh a$ (express the answer in terms of $o$):
- ReLU, $o = \max(0, a)$:

### Why gradients accumulate
Multivariable chain rule: if $a$ feeds into several nodes, $\partial L/\partial a$ is the sum over every path.

### Checking gradients numerically
Central differences, $\frac{f(x+\varepsilon) - f(x-\varepsilon)}{2\varepsilon}$, have error $O(\varepsilon^2)$. Show why.
