import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel('Lab Session Data (1).xlsx', sheet_name=2, usecols='B:AE')
df.replace('?', pd.NA, inplace=True)

numeric_cols = ['TSH','T3','TT4','T4U','FTI','TBG','age']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)

scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print(df[numeric_cols].head())
