"""A tiny neural-network library on top of Value."""
from __future__ import annotations

import random

from mg2t.micrograd.engine import Value


class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> list[Value]:
        return []


class Neuron(Module):
    def __init__(self, nin: int, nonlin: bool = True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: list) -> Value:
        # TODO: weighted sum plus bias, then tanh if self.nonlin
        raise NotImplementedError

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin: int, nout: int, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x: list):
        # TODO: apply every neuron; return a single Value if there is only one
        raise NotImplementedError

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, nin: int, nouts: list[int]):
        sizes = [nin] + nouts
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=(i != len(nouts) - 1))
            for i in range(len(nouts))
        ]

    def __call__(self, x: list):
        # TODO: pass x through each layer in turn
        raise NotImplementedError

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]
