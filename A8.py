import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
def deriv(x):
    return x*(1-x)
X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([0,0,0,1]).reshape(-1,1)
W1=np.random.randn(2,2)
W2=np.random.randn(2,1)
for epoch in range(1000):
    h=sigmoid(np.dot(X,W1))
    o=sigmoid(np.dot(h,W2))
    err=y-o
    if np.sum(err**2)<=0.002:
        break
    d2=err*deriv(o)
    d1=np.dot(d2,W2.T)*deriv(h)
    W2+=0.05*np.dot(h.T,d2)
    W1+=0.05*np.dot(X.T,d1)

print("Epochs:",epoch)