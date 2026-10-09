"""A scalar-valued autograd engine.

Each Value stores a number, its gradient, and how it was produced, so that
calling .backward() on an output fills in d(output)/d(node) for every node.

Your job: implement the methods marked TODO. The tests in tests/ check your
gradients against finite differences, so `pytest` tells you when you're done.
"""
from __future__ import annotations

import math


class Value:
    def __init__(self, data: float, _children: tuple = (), _op: str = "", label: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None  # set by each operation
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ---- Core operations: implement these ------------------------------

    def __add__(self, other) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        # TODO: create the output Value and define its _backward
        raise NotImplementedError

    def __mul__(self, other) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        # TODO
        raise NotImplementedError

    def __pow__(self, k: float) -> Value:
        assert isinstance(k, (int, float)), "only constant powers are supported"
        # TODO
        raise NotImplementedError

    def exp(self) -> Value:
        # TODO
        raise NotImplementedError

    def tanh(self) -> Value:
        # TODO
        raise NotImplementedError

    def relu(self) -> Value:
        # TODO
        raise NotImplementedError

    def backward(self) -> None:
        """Fill in .grad for every node that this Value depends on."""
        # TODO: order the graph so each node comes after its children,
        # set this node's grad, then call _backward in reverse order.
        raise NotImplementedError

    # ---- Derived operations: already done, built from the core ones ----

    def __neg__(self) -> Value:
        return self * -1

    def __radd__(self, other) -> Value:
        return self + other

    def __sub__(self, other) -> Value:
        return self + (-other)

    def __rsub__(self, other) -> Value:
        return other + (-self)

    def __rmul__(self, other) -> Value:
        return self * other

    def __truediv__(self, other) -> Value:
        return self * other**-1

    def __rtruediv__(self, other) -> Value:
        return other * self**-1


# `math` is imported for you; you'll likely need it.
_ = math
