import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
def load_dataset(file_path):
    df = pd.read_excel(file_path)
    return df

def processing_data(df):
    # Remove class label = 3 to keep only two classes
    df1 = df[df["label"] != 3]
    return df1

def training_testing_dataset(df):
    X = df.drop(columns=["label"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42 #ensures that the traning and test data produced is always same. controls reproducability
    )
    return X_train, X_test, y_train, y_test

def main():
    file_path = "Coherence_bert_cls_embeddings.xlsx"
    df = load_dataset(file_path)
    df_processed = processing_data(df)
    X_train, X_test, y_train, y_test = training_testing_dataset(df_processed)
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])
    print("Classes present:", y_train.unique())


main()
