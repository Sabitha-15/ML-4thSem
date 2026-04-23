import pandas as pd
import numpy as np
df=pd.read_csv("combined_eeg_dataset.csv")
df=df.drop(columns=["Subject"])
X=df.drop(columns=["label"]).values
y=df["label"].values
X=(X-X.mean())/X.std()
def summation(x,w):
    return np.dot(x,w[1:]) + w[0]
def sigmoid(x):
    return 1/(1+np.exp(-x))
def error(t,o):
    return t-o
def train(X,y,w):
    for epoch in range(1000):
        total=0
        for i in range(len(X)):
            net=summation(X[i],w)
            out=sigmoid(net)
            err=error(y[i],out)

            w[1:] += 0.05*err*X[i]
            w[0] += 0.05*err

            total += err**2
        if total<=0.002:
            break
    return epoch+1
w=np.random.randn(X.shape[1]+1)
print("Epochs:", train(X,y,w))