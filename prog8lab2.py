import pandas as pd
import numpy as np

df = pd.read_excel('Lab Session Data (1).xlsx', sheet_name=2, usecols='B:AE')
df.replace('?', pd.NA, inplace=True)

numeric_cols = ['TSH','T3','TT4','T4U','FTI','TBG','age']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

binary_cols = [col for col in df.select_dtypes(include='object').columns
               if df[col].dropna().isin(['t','f']).all()]
df[binary_cols] = df[binary_cols].applymap(lambda x: 1 if x=='t' else (0 if x=='f' else np.nan))


for col in numeric_cols:
    col_data = df[col]
    q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1-1.5*iqr, q3+1.5*iqr
    if ((col_data<lower)|(col_data>upper)).sum()==0:
        df[col].fillna(col_data.mean(), inplace=True)
    else:
        df[col].fillna(col_data.median(), inplace=True)

for col in binary_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing values filled.")
