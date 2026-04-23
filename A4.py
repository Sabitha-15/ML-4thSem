import numpy as np
import matplotlib.pyplot as plt
def summation(x,w):
    return np.dot(x,w[1:]) + w[0]
def step(x):
    return 1 if x>=0 else 0
def error(t,o):
    return t-o
def train(X,y,w,lr):
    for epoch in range(1000):
        total=0
        for i in range(len(X)):
            net=summation(X[i],w)
            out=step(net)
            err=error(y[i],out)

            w[1:] += lr*err*X[i]
            w[0] += lr*err

            total += err**2
        if total<=0.002:
            break
    return epoch+1

X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([0,0,0,1])
lrs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
epochs=[]
for lr in lrs:
    w=np.array([10.0,0.2,-0.75])
    epochs.append(train(X,y,w,lr))
print(list(zip(lrs,epochs)))
plt.plot(lrs,epochs)
plt.show()