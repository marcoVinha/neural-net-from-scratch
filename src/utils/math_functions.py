import numpy as np

from src.utils.base import Module, Tensor


class Sigmoid(Module):
    def __call__(self, X):
        out = Tensor(1 / (1 + np.exp(-X.data)))
        out._prev = {X}

        def _backward():
            X.grad += out.grad * (out.data * (1 - out.data))

        out._backward = _backward
        return out


class Softmax(Module):
    def __call__(self, X):
        exps = np.exp(X.data - np.max(X.data, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        out = Tensor(probs)
        out._prev = {X}

        def _backward():
            # Jacobian matrix for softmax gradient
            batch_size, n_classes = out.data.shape
            for i in range(batch_size):
                p = out.data[i].reshape(-1, 1)
                jacobian = np.diagflat(p) - p @ p.T
                X.grad[i] += out.grad[i] @ jacobian

        out._backward = _backward
        return out
