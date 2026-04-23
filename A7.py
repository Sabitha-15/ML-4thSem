import pandas as pd
import numpy as np
df=pd.read_csv("combined_eeg_dataset.csv")
df=df.drop(columns=["Subject"])
X=df.drop(columns=["label"]).values
y=df["label"].values
Xb=np.c_[np.ones(X.shape[0]),X]
w=np.linalg.pinv(Xb).dot(y)
pred=np.dot(Xb,w)
pred=np.where(pred>=0.5,1,0)
print("Accuracy:", np.mean(pred==y))