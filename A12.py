import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np
df=pd.read_csv("combined_eeg_dataset.csv")
df=df.drop(columns=["Subject"])
X=df.drop(columns=["label"]).values
y=df["label"].values
clf=MLPClassifier(hidden_layer_sizes=(10,),max_iter=1000)
clf.fit(X,y)
pred=clf.predict(X)
print("Accuracy:",np.mean(pred==y))