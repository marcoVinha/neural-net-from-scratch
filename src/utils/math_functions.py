import numpy as np

from src.utils.base import Module, Tensor


class Sigmoid(Module):
    def __call__(self, X):
        out = Tensor(1 / (1 + np.exp(-1 * X.data)))
        out._prev = {X}

        def _backward():
            X.grad += out.data * (1 - out.data) * out.grad

        out._backward = _backward
        return out


class Softmax(Module):
    def __call__(self, X):
        exps = np.exp(x.data - np.max(x.data, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        out = Tensor(probs)
        out._prev = {X}

        def _backward():
            # Jacobian matrix for softmax gradient
            batch_size, n_classes = out.data.shape
            for i in range(batch_size):
                p = out.data[i].reshape(-1, 1)
                jacobian = np.diagflat(p) - p @ p.T
                X.grad[i] += jacobian @ out.grad[i]

        out._backward = _backward
        return out


class CrossEntropy(Module):
    def __call__(self, probs, y):
        eps = 1e-15
        loss = -np.sum(y * np.log(probs + eps)) / probs.data.shape[0]
        out = Tensor(loss)
        out._prev = {probs}

        def _backward():
            probs.grad += (-y / (probs.data + eps)) / probs.data.shape[0]

        out._backward = _backward
        return out
