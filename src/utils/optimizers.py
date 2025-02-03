import numpy as np


class SGDOptimizer:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        """Reset gradients for all parameters"""

        for param in self.params:
            param.grad = np.zeros_like(param.data)

    def step(self):
        """Perform a single optmizing step"""

        for param in self.params:
            param.data -= self.lr * param.grad
