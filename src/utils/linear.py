import numpy as np

from src.utils.base import Module, Tensor


class Linear(Module):
    def __init__(self, input_size, output_size):
        self.W = Tensor(np.random.randn(input_size, output_size), requires_grad=True)
        self.b = Tensor(np.zeros((1, output_size)), requires_grad=True)

    def __call__(self, X):
        out = Tensor(X.data @ self.W.data + self.b.data)
        out._prev = {self.W, self.b}

        def _backward():
            self.W.grad += X.data.T @ out.grad
            self.b.grad += np.sum(out.grad, axis=0, keepdims=True)

        out._backward = _backward
        return out

    def parameters(self):
        return [self.W, self.b]
