import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
y = np.array([
    [1,0],
    [1,0],
    [1,0],
    [0,1]
])
X_bias = np.c_[np.ones(X.shape[0]), X]
W = np.random.randn(X_bias.shape[1], 2)
for epoch in range(1000):
    output = sigmoid(np.dot(X_bias, W))
    error = y - output
    W += 0.05 * np.dot(X_bias.T, error)
    if np.sum(error**2) <= 0.002:
        break
print("Epochs to converge:", epoch)
print("Final Output:\n", output)