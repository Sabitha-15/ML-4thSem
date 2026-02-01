import numpy as np
import pandas as pd
def load_dataset(file_path):
    df=pd.read_excel(
        file_path,
        usecols=lambda x: x != "B"
    )
    
    return df

def processing_paragraphs(df):
    feature_Cols=[col for col in df.columns if col.startswith("f")]
    groups=df.groupby("para_id")
    paragraph_vectors=[]
    for para_id,group in groups:
        unique_sentences=group[feature_Cols].drop_duplicates()
        paragraph_embedding=unique_sentences.mean(axis=0)
        label=group["label"].iloc[0] # in that particular group take the column label and take the first element of that 
                                     #column since all the labels with in a group are same
        paragraph_vectors.append([para_id]+paragraph_embedding.tolist()+[label])
    
    para_df=pd.DataFrame(paragraph_vectors,columns=["para_id"]+feature_Cols+["label"]) #creating paragraph level dataframe
    class1=para_df.loc[para_df['label']==1][feature_Cols].to_numpy() #converts pandas dataframe to numpy array
    class2=para_df.loc[para_df['label']==2][feature_Cols].to_numpy() #we can also use .values to convert from df to array 
    class3=para_df.loc[para_df['label']==3][feature_Cols].to_numpy() #recommended one is to_numpy()
    
    return class1,class2,class3

def mean_each_class(class1,class2,class3):
    centroid_class1=class1.mean(axis=0)
    centroid_class2=class2.mean(axis=0)
    centroid_class3=class3.mean(axis=0)
    
    return centroid_class1,centroid_class2,centroid_class3

def std_each_calss(class1,class2,class3):
    std_class1=class1.std(axis=0)
    std_class2=class2.std(axis=0)
    std_class3=class3.std(axis=0)
    
    return std_class1,std_class2,std_class3

def spread_each_class(class1,class2,class3):
    c1,c2,c3=mean_each_class(class1,class2,class3)
    interclassspread_c1_c2=np.linalg.norm(c1-c2)
    interclassspread_c2_c3=np.linalg.norm(c3-c2)
    interclassspread_c1_c3=np.linalg.norm(c1-c3)
    
    return interclassspread_c1_c2,interclassspread_c2_c3,interclassspread_c1_c3

def main():
    file_path="Coherence_bert_cls_embeddings.xlsx"
    df=load_dataset(file_path)
    class1,class2,class3=processing_paragraphs(df)
    mean1,mean2,mean3=mean_each_class(class1,class2,class3)
    std1,std2,std3=std_each_calss(class1,class2,class3)
    print("centeroid of class label 1: ",mean1)
    print("centeroid of class label 2: ",mean2)
    print("centeroid of class label 3: ",mean3)
    print("intraclass spread for class1: ",std1)
    print("intraclass spread for class2: ",std2)
    print("intraclass spread for class3: ",std3)
    print(mean1.shape)
    inter1,inter2,inter3=spread_each_class(class1,class2,class3)
    print("interclass spread between class 1 and 2: ",inter1)
    print("interclass spread between class 3 and 2: ",inter2)
    print("interclass spread between class 1 and 3: ",inter3)

main()
        
    
    
    
