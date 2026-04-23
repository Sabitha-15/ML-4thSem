import numpy as np
import matplotlib.pyplot as plt
def summation(x,w):
    return np.dot(x,w[1:]) + w[0]
def step(x):
    return 1 if x>=0 else 0
def error(t,o):
    return t-o
def train(X,y,w,lr):
    errors=[]
    for epoch in range(1000):
        total=0
        for i in range(len(X)):
            net=summation(X[i],w)
            out=step(net)
            err=error(y[i],out)
            w[1:] += lr*err*X[i]
            w[0] += lr*err
            total += err**2
        errors.append(total)
        if total<=0.002:
            break
    return w,errors,epoch+1

X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([0,0,0,1])
w=np.array([10.0,0.2,-0.75])
wf,err,ep=train(X,y,w,0.05)
print("Weights:",wf)
print("Epochs:",ep)
plt.plot(err)
plt.show()