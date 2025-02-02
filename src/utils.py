import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy(y_true, y_pred):
    # Avoid log(0) by clipping predictions to [epsilon, 1 - epsilon]
    epsilon = 1e-10
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

    # Compute loss
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / y_true.shape[0]
    return loss
