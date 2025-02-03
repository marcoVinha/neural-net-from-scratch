from src.utils.base import Module, Tensor


class CrossEntropy(Module):
    def __call__(self, probs, y):
        eps = 1e-15
        loss = -1 * np.sum(y * np.log(probs.data + eps)) / probs.data.shape[0]
        out = Tensor(loss)
        out._prev = {probs}

        def _backward():
            probs.grad += (-y / (probs.data + eps)) / probs.data.shape[0]

        out._backward = _backward
        return out
