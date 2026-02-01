import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
def load_dataset(file_path):
    df = pd.read_excel(file_path)
    return df

def processing_data(df):
    df1 = df[df["label"] != 3]
    return df1

def training_testing_dataset(df, test_size=0.3, random_state=42):
    X = df.drop(columns=["label"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def train_knn_sklearn(X_train, y_train, k=3):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    return neigh

def test_knn_accuracy(neigh, X_test, y_test):
    return neigh.score(X_test, y_test)

def knn_predict(neigh, X_test):
    return neigh.predict(X_test)

def custom_knn_predict(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_vect in X_test.to_numpy():
        distances = np.linalg.norm(X_train.to_numpy() - test_vect, axis=1)
        nn_indices = distances.argsort()[:k]
        nn_labels = y_train.to_numpy()[nn_indices]
        values, counts = np.unique(nn_labels, return_counts=True)
        y_pred.append(values[counts.argmax()])
    return np.array(y_pred)

def custom_knn_accuracy(y_test, y_pred):
    return np.mean(y_test.to_numpy() == y_pred)

def knn_accuracy_plot(X_train, y_train, X_test, y_test, max_k=11):
    acc_list = []
    k_values = list(range(1, max_k+1))
    for k in k_values:
        y_pred = custom_knn_predict(X_train, y_train, X_test, k)
        acc = custom_knn_accuracy(y_test, y_pred)
        acc_list.append(acc)
    plt.plot(k_values, acc_list, marker='o')
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Accuracy")
    plt.title("kNN Accuracy vs k")
    plt.grid(True)
    plt.show()
    
def main():
    file_path = "Coherence_bert_cls_embeddings.xlsx"
    df = load_dataset(file_path)
    df_processed = processing_data(df)
    X_train, X_test, y_train, y_test = training_testing_dataset(df_processed)
    neigh = train_knn_sklearn(X_train, y_train, k=3)
    accuracy_sklearn = test_knn_accuracy(neigh, X_test, y_test)
    print("Sklearn kNN Accuracy (k=3):", accuracy_sklearn)
    y_pred_sklearn = knn_predict(neigh, X_test)
    print("Predictions (first 10):", y_pred_sklearn[:10])
    y_pred_custom = custom_knn_predict(X_train, y_train, X_test, k=3)
    accuracy_custom = custom_knn_accuracy(y_test, y_pred_custom)
    print("Custom kNN Accuracy (k=3):", accuracy_custom)
    y_pred_nn = custom_knn_predict(X_train, y_train, X_test, k=1)
    accuracy_nn = custom_knn_accuracy(y_test, y_pred_nn)
    print("NN Classifier Accuracy (k=1):", accuracy_nn)
    knn_accuracy_plot(X_train, y_train, X_test, y_test, max_k=11)

main()
