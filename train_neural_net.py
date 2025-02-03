import numpy as np

from src.utils.base import Sequential, Tensor
from src.utils.linear import Linear
from src.utils.losses import CrossEntropyLoss
from src.utils.math_functions import Sigmoid, Softmax
from src.utils.optimizers import SGDOptimizer


np.random.seed(0)

input_size = 2
hidden_size = 2
output_size = 2

# Synthetic data (3 samples, 2 features)
X = np.array([[0.5, 0.1], [0.9, 0.8], [0.2, 0.3]])

# Labels
y = np.array([[1, 0], [0, 1], [1, 0]])

model = Sequential(
    Linear(input_size, hidden_size),
    Sigmoid(),
    Linear(hidden_size, output_size),
)

softmax = Softmax()
loss_fn = CrossEntropyLoss()

optimizer = SGDOptimizer(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()

    logits = model(Tensor(X))
    probs = softmax(logits)
    loss = loss_fn(probs, y)

    loss.grad = 1.0

    visited = set()
    def topo_sort(node):
        if node not in visited:
            visited.add(node)
            for prev_node in node._prev:
                topo_sort(prev_node)
            topo_order.append(node)
    topo_order = []
    breakpoint()
    topo_sort(loss)
    
    print(topo_order)

    for node in reversed(topo_order):
        print(node.__class__.__name__)
        node._backward()

    optimizer.step()

    loss_history.append(loss.data)
    
    # if epoch % 100 == 0:
    print(f"Epoch {epoch}, Loss: {loss.data}")
