import numpy as np
from utils import cross_entropy, sigmoid, softmax


np.random.seed(0)

input_size = 2
hidden_size = 2
output_size = 2

# Synthetic data (3 samples, 2 features)
X = np.array([[0.5, 0.1], [0.9, 0.8], [0.2, 0.3]])

# Labels
y = np.array([[1, 0], [0, 1], [1, 0]])

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Training loop
epochs = 1000
loss_history = []

for epoch in range(epochs):
    # For each layer, the forward pass is:
    #   Z_n = X_n @ W_n + b_n
    #   A_n = activation(Z_n)

    # Forward pass for the first hidden layer
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)

    # Forward pass for the second hidden layer
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    # As we only have two hidden layers, the
    # output of the network is A2, which is
    # why the last activation function is a
    # softmax function.
    y_pred = A2

    # Compute loss after the forward pass
    curr_loss = cross_entropy(y, y_pred)
    loss_history.append(curr_loss)

    # The general formula for the gradients
    # of the weights and biases is:
    #   dLossW_l = A_(l-1)^T @ dLossZ_l / N
    #   dLossb_l = sum(dLossZ_l) / N
    #
    # We need to calculate the following gradients:
    #   dLossW2, dLossb2, dLossW1, dLossb1
    # in that order.
    #
    # Here are the expressions for each layer:
    #   dLossW2 = (1/N) * A1.T @ dLossZ2
    #   dLossb2 = (1/N) * sum(dLossZ2)
    #   dLossW1 = (1/N) * X.T @ dLossZ1
    #   dLossb1 = (1/N) * sum(dLossZ1)

    # Now, for the backward pass...
    # First we calculate the terms that don't
    # depend on anything, which are the gradients
    # of the loss with respect to the activations
    # of the second hidden layer and the first
    # hidden layer.

    # Derivative of the loss with respect to the
    # activations of the second hidden layer,
    # which is the output of the softmax (dLossZ2
    # is equivalent to dA2Z2).
    dLossZ2 = y_pred - y

    # Derivative of the loss with respect to the
    # activations of the first hidden layer, which
    # is the output of the sigmoid (A1).
    dLossA1 = dLossZ2 @ W2.T

    # Derivative of the loss with respect to the
    # weighted sum of the first hidden layer,
    # which is the input to the sigmoid (Z1).
    dLossZ1 = dLossA1 * (A1 * (1 - A1))

    # With this info, we can calculate the gradients
    # of the loss with respect to the weights and
    # biases of each layer.

    # The gradient of the loss with respect to the
    # weights and biases of the second hidden layer
    # (dLossW2 and dLossb2) are calculated by the
    # formula we already showed:
    #   dLossW_l = A_(l-1)^T @ dLossZ_l / N
    #   dLossb_l = sum(dLossZ_l) / N
    dLossW2 = (A1.T @ dLossZ2) / X.shape[0]
    dLossb2 = np.sum(dLossZ2, axis=0, keepdims=True) / X.shape[0]

    # The gradient of the loss with respect to the
    # weights and biases of the first hidden layer
    # (dLossW1 and dLossb1) are calculated in the
    # same way. Now, instead of an "A0", we have
    # the input data X.
    dLossW1 = X.T @ dLossZ1 / X.shape[0]
    dLossb1 = np.sum(dLossZ1, axis=0, keepdims=True) / X.shape[0]

    # The learning rate is a hyperparameter that
    # controls how much we update the weights and
    # biases at each step. We will use a fixed
    # learning rate of 0.01.
    learning_rate = 0.01

    # Now that we have the gradients for weights
    # and biases of each layer, we can update the
    # weights and biases themselves, by applying:
    #   W_l -= learning_rate * dLossW_l
    W1 -= learning_rate * dLossW1
    b1 -= learning_rate * dLossb1
    W2 -= learning_rate * dLossW2
    b2 -= learning_rate * dLossb2

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {curr_loss}")


import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()
