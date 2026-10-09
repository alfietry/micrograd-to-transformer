"""Checks the autograd engine against finite differences (and PyTorch if installed)."""
import math

import pytest

from mg2t.micrograd.engine import Value

EPS = 1e-6


def numerical_grads(f, xs):
    """Central-difference estimate of df/dx_i for each input."""
    grads = []
    for i in range(len(xs)):
        up = list(xs); up[i] += EPS
        dn = list(xs); dn[i] -= EPS
        grads.append((f(*[Value(v) for v in up]).data - f(*[Value(v) for v in dn]).data) / (2 * EPS))
    return grads


def check(f, xs, tol=1e-4):
    vals = [Value(v) for v in xs]
    out = f(*vals)
    out.backward()
    for v, g in zip(vals, numerical_grads(f, xs)):
        assert v.grad == pytest.approx(g, abs=tol)


def test_forward_values():
    a, b = Value(2.0), Value(-3.0)
    assert (a + b).data == -1.0
    assert (a * b).data == -6.0
    assert (a**3).data == 8.0
    assert a.exp().data == pytest.approx(math.exp(2.0))
    assert b.tanh().data == pytest.approx(math.tanh(-3.0))
    assert b.relu().data == 0.0 and a.relu().data == 2.0


def test_add_mul():
    check(lambda a, b, c: a * b + c, [2.0, -3.0, 10.0])


def test_pow_and_division():
    check(lambda a, b: a**2 / b + b**-1.5, [1.7, 2.3])


def test_exp_tanh():
    check(lambda a, b: (a * b).tanh() + (a - b).exp(), [0.4, -0.9])


def test_relu():
    check(lambda a, b: (a * b).relu() + (a - b).relu(), [1.5, -2.0])


def test_reused_node_accumulates_gradient():
    # d/da (a + a) = 2: gradients must add up, not overwrite
    a = Value(3.0)
    (a + a).backward()
    assert a.grad == pytest.approx(2.0)


def test_deep_expression():
    def f(a, b):
        c = a * b + b**3
        d = (c * 2 + 1).tanh()
        e = d * a - c / (b * b + 1)
        return e * e + e.relu()
    check(f, [-0.6, 1.3])


def test_matches_pytorch():
    torch = pytest.importorskip("torch")
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()

    xt = torch.tensor([-4.0], dtype=torch.double, requires_grad=True)
    zt = 2 * xt + 2 + xt
    qt = zt.relu() + zt * xt
    ht = (zt * zt).relu()
    yt = ht + qt + qt * xt
    yt.backward()

    assert y.data == pytest.approx(yt.item())
    assert x.grad == pytest.approx(xt.grad.item())
