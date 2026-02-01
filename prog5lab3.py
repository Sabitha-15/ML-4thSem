import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
def load_dataset(file_path):
    df = pd.read_excel(
        file_path,
        usecols="D:ACQ"
    )
    return df

def processing_Dataset(df):
    df1 = df.iloc[:, [0, 1]]
    v1 = df1.iloc[:, 0].to_numpy()
    v2 = df1.iloc[:, 1].to_numpy()
    return v1, v2

def custom_minkowski_distance(v1, v2):
    custom_distances = []
    for p in range(1, 11):
        dist = np.sum(np.abs(v1 - v2) ** p) ** (1 / p)
        custom_distances.append(dist)
    return custom_distances

def scipy_minkowski_distance(v1, v2):
    scipy_distances = []
    for p in range(1, 11):
        dist = minkowski(v1, v2, p)
        scipy_distances.append(dist)
    return scipy_distances

def plot_minkowski_comparison(custom_dist, scipy_dist, p_values):
    plt.figure()
    plt.plot(p_values, custom_dist, marker='o')
    plt.plot(p_values, scipy_dist, marker='x')
    plt.xlabel("p value")
    plt.ylabel("Minkowski Distance")
    plt.title("Custom vs SciPy Minkowski Distance")
    plt.legend(["Custom Function", "SciPy Function"])
    plt.show()
    
def print_comparison(custom_dist, scipy_dist, p_values):
    print("p-value | Custom Minkowski | SciPy Minkowski")
    for p, c, s in zip(p_values, custom_dist, scipy_dist):
        print(f"{p:^7} | {c:^16.6f} | {s:^14.6f}")

def main():
    file_path = "Coherence_bert_cls_embeddings.xlsx"
    df = load_dataset(file_path)
    v1, v2 = processing_Dataset(df)
    p_values = range(1, 11)
    custom_dist = custom_minkowski_distance(v1, v2)
    scipy_dist = scipy_minkowski_distance(v1, v2)
    print_comparison(custom_dist, scipy_dist, p_values)
    plot_minkowski_comparison(custom_dist, scipy_dist, p_values)

main()
