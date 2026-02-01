import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def load_dataset(file_path):
    df=pd.read_excel(
        file_path,
        usecols="D:ACQ"
    )
    
    return df

def processing_Dataset(df):
    df1=df.iloc[:,[0,1]]
    v1=df1.iloc[:,0].to_numpy()
    v2=df1.iloc[:,1].to_numpy()
    
    return v1,v2

def calculate_minkowski_distance(v1,v2):
    minkowski_dist_1to10=[]
    for i in range(1,11,1):
        minkowski=0
        minkowski=np.sum(np.abs(v1-v2)**i)**(1/i)
        minkowski_dist_1to10.append(minkowski)
        
    return minkowski_dist_1to10

def plotting_the_graph(minkowski_dist_1to10,values_range):
    plt.figure()
    plt.plot(values_range, minkowski_dist_1to10, marker='o')
    plt.xlabel("values range")
    plt.ylabel("Minkowski Distance")
    plt.title("Minkowski Distance vs p")
    plt.show()


def main():
    file_path="Coherence_bert_cls_embeddings.xlsx"
    df=load_dataset(file_path)
    v1,v2=processing_Dataset(df)
    minkowski_dist_1to10=calculate_minkowski_distance(v1,v2)
    print(minkowski_dist_1to10)
    values_range=range(1,11,1)
    plotting_the_graph(minkowski_dist_1to10,values_range)
    
main()
    