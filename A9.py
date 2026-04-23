import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x):
    return x * (1 - x)
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
y = np.array([0,1,1,0]).reshape(-1,1)
np.random.seed(0)
W1 = np.random.randn(2, 2)
W2 = np.random.randn(2, 1)
for epoch in range(1000):
    hidden = sigmoid(np.dot(X, W1))
    output = sigmoid(np.dot(hidden, W2))
    error = y - output
    if np.sum(error**2) <= 0.002:
        break
    d2 = error * sigmoid_deriv(output)
    d1 = np.dot(d2, W2.T) * sigmoid_deriv(hidden)
    W2 += 0.05 * np.dot(hidden.T, d2)
    W1 += 0.05 * np.dot(X.T, d1)
print("Epochs to converge:", epoch)
print("Final Output:\n", output)