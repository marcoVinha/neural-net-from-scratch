import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self.prev = set()
        self._backward = lambda: None


class Module:
    def parameters(self):
        return []


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, X):
        for layer in self.layers:
            x = layer(X)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params
