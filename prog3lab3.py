import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def load_dataset(file_path):
    df=pd.read_excel(
        file_path
    )
    
    return df

def plotting_function(df,feature_name):
    
    feature_values = df[feature_name].values
    mean_value = np.mean(feature_values)
    variance_value = np.var(feature_values)
    counts, bin_edges = np.histogram(feature_values, bins=20)
    plt.figure()
    plt.hist(feature_values, bins=20)
    plt.xlabel(f"Values of embedding feature {feature_name}")
    plt.ylabel("Number of sentences")
    plt.title(f"Histogram of {feature_name}")
    plt.show()
    
    return mean_value,variance_value

def main():
    file_path="Coherence_bert_cls_embeddings.xlsx"
    df=load_dataset(file_path)
    print(df.columns)

    feature_name=input("enter a feature name: ")
    mean_value,variance_value=plotting_function(df,feature_name)
    print(f"Mean of {feature_name}: {mean_value}")
    print(f"Variance of {feature_name}: {variance_value}")

main()