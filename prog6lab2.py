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

categorical_cols = [col for col in df.select_dtypes(include='object').columns if col not in binary_cols]
df_encoded = pd.get_dummies(df, columns=categorical_cols, dummy_na=True)


vec1 = df_encoded.iloc[0].to_numpy(dtype=float)
vec2 = df_encoded.iloc[1].to_numpy(dtype=float)


mask = ~np.isnan(vec1) & ~np.isnan(vec2)
vec1, vec2 = vec1[mask], vec2[mask]

cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print(f"Cosine Similarity (first 2 rows): {cos_sim:.4f}")
