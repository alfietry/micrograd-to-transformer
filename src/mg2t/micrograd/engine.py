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

    # ---- Core operations  ------------------------------

    def __add__(self, other) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k: float) -> Value:
        assert isinstance(k, (int, float)), "only constant powers are supported"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += (k * self.data ** (k - 1)) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        out = Value(math.tanh(self.data), (self,), "tanh")

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Fill in .grad for every node that this Value depends on."""
        topo = []
        seen = set()

        def build_topo(v: Value):
            stack = [(v, False)]
            while stack:
                node, expanded = stack.pop()
                if expanded:
                    topo.append(node)
                    continue
                if node in seen:
                    continue
                seen.add(node)
                stack.append((node, True))
                for child in reversed(tuple(node._prev)):
                    if child not in seen:
                        stack.append((child, False))

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

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
