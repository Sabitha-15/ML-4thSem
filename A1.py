import numpy as np
def summation(x, weights):
    return np.dot(x, weights[1:]) + weights[0]
def step(x):
    return 1 if x >= 0 else 0
def bipolar_step(x):
    return 1 if x >= 0 else -1
def sigmoid(x):
    return 1/(1+np.exp(-x))
def tanh(x):
    return np.tanh(x)
def relu(x):
    return max(0, x)
def leaky_relu(x):
    return x if x > 0 else 0.01*x
def error(t, o):
    return t - o

x = np.array([1, 2])
weights = np.array([0.5, 0.3, -0.2])  # w0, w1, w2
net = summation(x, weights)
print("Summation Output:", net)
print("Step:", step(net))
print("Bipolar Step:", bipolar_step(net))
print("Sigmoid:", sigmoid(net))
print("TanH:", tanh(net))
print("ReLU:", relu(net))
print("Leaky ReLU:", leaky_relu(net))
target = 1
output = step(net)
print("Error:", error(target, output))