import numpy as np
def summation(x,w):
    return np.dot(x,w[1:]) + w[0]
def error(t,o):
    return t-o
def bipolar(x):
    return 1 if x>=0 else -1
def sigmoid(x):
    return 1/(1+np.exp(-x))
def relu(x):
    return max(0,x)
def train(X,y,w,lr,act):
    for epoch in range(1000):
        total=0
        for i in range(len(X)):
            net=summation(X[i],w)
            out=act(net)
            err=error(y[i],out)

            w[1:] += lr*err*X[i]
            w[0] += lr*err

            total += err**2
        if total<=0.002:
            break
    return epoch+1

X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([0,0,0,1])
acts={"Bipolar":bipolar,"Sigmoid":sigmoid,"ReLU":relu}
for name,f in acts.items():
    w=np.array([10.0,0.2,-0.75])
    print(name, train(X,y,w,0.05,f))