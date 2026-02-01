import pandas as pd
import numpy as np
import time
def load_dataset(file_path):
    df=pd.read_excel(
        file_path,
        usecols="D:E",
        nrows=6
    )
    
    return df

def feature_vectors(df):
    df1=df.iloc[:,0]
    df2=df.iloc[:,1]
    v1=df1.to_numpy()
    v2=df2.to_numpy()
    
    return v1,v2

def dot_product_norm(v1,v2):
    start1=time.perf_counter()
    dot_prod=v1.dot(v2)
    end1=time.perf_counter()
    start2=time.perf_counter()
    norm1=np.linalg.norm(v1)
    norm2=np.linalg.norm(v2)
    end2=time.perf_counter()
    time1=start1-end1
    time2=start2-end2
    
    return (time1,time2,norm1,norm2,dot_prod)

def dot_prod_norm_manual(v1,v2):
    start1=time.perf_counter()
    dot_prod=0
    norm_squaredv1=0
    norm_squaredv2=0
    for i,j in zip(v1,v2):
        dot_prod+=i*j
        norm_squaredv1+=i**2
        norm_squaredv2+=j**2
    norm_v1=np.sqrt(norm_squaredv1)
    norm_v2=np.sqrt(norm_squaredv2)
    end1=time.perf_counter()
    time_taken=start1-end1
    
    return time_taken,norm_v1,norm_v2,dot_prod

def main():
    file_path="Coherence_bert_cls_embeddings.xlsx"
    df=load_dataset(file_path)
    v1,v2=feature_vectors(df)
    time1,time2,norm1,norm2,dot_prod=dot_product_norm(v1,v2)
    time_taken,norm_v1,norm_v2,dot_prod1=dot_prod_norm_manual(v1,v2)
    print("norm of the vector v1 using norm function: ",norm1 )
    print("norm of the vector v1 using norm function: ",norm2 )
    print("dot product of vectors v1 and v2 using the numpy funciton: ",dot_prod )
    print("norm of the vector v1 using manual function: ",norm_v1 )
    print("norm of the vector v1 using manual function: ",norm_v2  )
    print("dot product of vectors v1 and v2 using the manual funciton: ",dot_prod1)
    time_taken_functions=time1+time2
    if time_taken_functions>time_taken:
        print("time taken by the numpy functions is more")
    else:
        print("time taken by the numpy functions is less")
        
main()

        
    
    
    


        
    
        
        
    


