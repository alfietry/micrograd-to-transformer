import random

from mg2t.micrograd.nn import MLP


def test_mlp_shapes_and_parameter_count():
    model = MLP(3, [4, 4, 1])
    assert len(model.parameters()) == (3 * 4 + 4) + (4 * 4 + 4) + (4 * 1 + 1)
    out = model([1.0, -2.0, 0.5])
    assert hasattr(out, "data")  # single output neuron returns one Value


def test_training_reduces_loss():
    random.seed(0)
    model = MLP(3, [4, 4, 1])
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]

    def loss():
        return sum((model(x) - y) ** 2 for x, y in zip(xs, ys))

    start = loss().data
    for _ in range(50):
        L = loss()
        model.zero_grad()
        L.backward()
        for p in model.parameters():
            p.data -= 0.05 * p.grad
    assert loss().data < 0.25 * start
