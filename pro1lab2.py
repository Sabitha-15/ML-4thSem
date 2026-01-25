import pandas as pd
import numpy as np
def load_data(file_path):
    data_frame = pd.read_excel(
        file_path,
        sheet_name=0,
        usecols="B:E"
    )
    return data_frame

def prepare_features_and_target(data_frame):
    features = data_frame.iloc[:, 0:3].to_numpy()
    target = data_frame.iloc[:, 3].to_numpy()
    return features, target

def calculate_coefficients(features, target):
    coefficients = np.linalg.pinv(features) @ target
    return coefficients

def classify_customer(data_frame):
    data_frame['Result'] = np.where(
        data_frame.iloc[:, 3] >= 200,
        'Rich',
        'Poor'
    )
    return data_frame

def main():

    df = load_data("Lab Session Data (1).xlsx")

    X, y = prepare_features_and_target(df)

    coefficients = calculate_coefficients(X, y)

    df = classify_customer(df)

    print("Features (X):\n", X)
    print("\nTarget (y):\n", y)
    print("\nCoefficients:\n", coefficients)
    print("\nFinal DataFrame:\n", df)

    df.to_excel("question1.xlsx", index=False)

main()
