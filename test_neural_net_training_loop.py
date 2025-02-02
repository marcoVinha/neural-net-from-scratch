import numpy as np
from src.utils import cross_entropy, sigmoid, softmax


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
    # For each layer `l`, the forward pass is,
    # basically, a matrix multiplication to
    # get the weighted sum of the inputs, and
    # then applying an activation function to
    # get the output of the layer. The general
    # formula for the forward pass is:
    #
    #   Z_l = A_(l-1) @ W_l + b_l
    #   A_l = activation(Z_l)
    #
    # where `A_(l-1)` is the output of the previous
    # activation layer (or the input `X`), `W_l`
    # is the weights of the layer, `b_l` is the
    # biases of the layer, and `activation` is
    # the activation function of the layer.

    # Forward pass for the first hidden layer:
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)

    # Forward pass for the second hidden layer:
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    # As we only have two hidden layers, the
    # output of the network is `A2`, which is
    # why the last activation function is a
    # softmax function (for this task, which
    # is a classification task).
    y_pred = A2

    # The loss is what we'll use to update the
    # weights and biases of the network. We'll
    # use the cross-entropy loss, which is a
    # common loss function for classification
    # tasks. The formula is:
    curr_loss = cross_entropy(y, y_pred)
    loss_history.append(curr_loss)

    # Now, for the backward pass...
    # The general formula for the gradients
    # of the weights and biases is:
    #
    #   dLoss/dW_l = A_(l-1)^T @ dLoss/dZ_l * (1 / N)
    #   dLoss/db_l = sum(dLoss/dZ_l) / N
    #
    # We need to calculate the following gradients:
    #
    #   dLoss/dW2, dLoss/db2, dLoss/dW1, dLoss/db1
    #
    # to update all weights and biases.
    #
    # Here are the expressions for each layer:
    #   dLossd/dW2 = (1/N) * A1.T @ dLoss/dZ2
    #   dLoss/db2 = (1/N) * sum(dLoss/dZ2)
    #   dLoss/dW1 = (1/N) * X.T @ dLoss/dZ1
    #   dLoss/db1 = (1/N) * sum(dLoss/dZ1)
    #
    # First we calculate the terms that don't
    # depend on anything, which are the gradients
    # of the loss with respect to the activations
    # of the second hidden layer and the first
    # hidden layer.

    # Derivative of the loss with respect to the
    # outputs of the second hidden layer, which
    # is the output of the softmax.
    #
    # Note: we skip the calculation of `dLoss/dA2`
    # because, when calculating `dLoss/dZ2`,
    # **because of the softmax function in combination
    # with the cross-entropy loss**, terms cancel out:
    #
    #   dLoss/dZ2 = dLoss/dA2 * dA2/dZ2
    #
    #   dLoss/dA2 = -y / A2
    #   dA2/dZ2 = A2 * (1 - A2)
    #
    #   dLoss/dZ2 = (-y / A2) * A2 * (1 - A2)
    #   dLoss/dZ2 = -y * (1 - A2)
    #   dLoss/dZ2 = A2 - y = y_pred - y
    #
    # This is a very important property of the
    # softmax function in combination with the
    # cross-entropy loss, which makes the
    # calculation of the gradient of the loss
    # with respect to the last hidden layer
    # much simpler.
    dLossdZ2 = y_pred - y

    # Derivative of the loss with respect to the
    # activations of the first hidden layer, which
    # is the output of the sigmoid (`A1`).
    dLossdA1 = dLossdZ2 @ W2.T

    # Derivative of the loss with respect to the
    # weighted sum of the first hidden layer (`Z1`),
    # which is the input to the sigmoid.
    dLossdZ1 = dLossdA1 * (A1 * (1 - A1))

    # With this info, we can calculate the gradients
    # of the loss with respect to the weights and
    # biases of each layer.

    # The gradient of the loss with respect to the
    # weights and biases of the second hidden layer
    # (`dLoss/dW2` and `dLoss/db2`) are calculated by the
    # formula we already showed:
    #
    #   dLoss/dW_l = A_(l-1)^T @ dLoss/dZ_l * (1 / N)
    #   dLoss/db_l = sum(dLoss/dZ_l) / N
    #
    # Applying this formula to the second hidden
    # layer, we get:
    dLossdW2 = (A1.T @ dLossdZ2) / X.shape[0]
    dLossdb2 = np.sum(dLossdZ2, axis=0, keepdims=True) / X.shape[0]

    # The gradient of the loss with respect to the
    # weights and biases of the first hidden layer
    # (`dLoss/dW1` and `dLoss/db1`) are calculated in the
    # same way. Now, instead of an `A0`, we have
    # the input data `X`.
    dLossdW1 = X.T @ dLossdZ1 / X.shape[0]
    dLossdb1 = np.sum(dLossdZ1, axis=0, keepdims=True) / X.shape[0]

    # The learning rate is a hyperparameter that
    # controls how much we update the weights and
    # biases at each step.
    learning_rate = 0.01

    # Now that we have the gradients for weights
    # and biases of each layer, we can update the
    # weights and biases themselves, by applying:
    #
    #   W_l -= learning_rate * dLoss/dW_l
    #   b_l -= learning_rate * dLoss/db_l
    #
    # for each layer.
    W1 -= learning_rate * dLossdW1
    b1 -= learning_rate * dLossdb1
    W2 -= learning_rate * dLossdW2
    b2 -= learning_rate * dLossdb2

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {curr_loss}")


print(f"Final loss: {curr_loss}")
