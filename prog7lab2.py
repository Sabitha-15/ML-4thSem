import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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


def jaccard_smc(row1, row2):
    f00=f01=f10=f11=0
    for i,j in zip(row1,row2):
        if i==1 and j==1: f11+=1
        elif i==1 and j==0: f10+=1
        elif i==0 and j==1: f01+=1
        else: f00+=1
    jc = f11/(f11+f10+f01) if (f11+f10+f01)!=0 else 0
    smc = (f11+f00)/(f11+f10+f01+f00)
    return jc, smc

def cosine_similarity(row1, row2):
    vec1 = row1.to_numpy(dtype=float)
    vec2 = row2.to_numpy(dtype=float)
    mask = ~np.isnan(vec1) & ~np.isnan(vec2)
    vec1, vec2 = vec1[mask], vec2[mask]
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

n = 20
jc_matrix = np.zeros((n,n))
smc_matrix = np.zeros((n,n))
cos_matrix = np.zeros((n,n))
df_binary = df_encoded[binary_cols].iloc[:n]
df_all = df_encoded.iloc[:n]

for i in range(n):
    for j in range(n):
        jc, smc = jaccard_smc(df_binary.iloc[i], df_binary.iloc[j])
        cos = cosine_similarity(df_all.iloc[i], df_all.iloc[j])
        jc_matrix[i,j] = jc
        smc_matrix[i,j] = smc
        cos_matrix[i,j] = cos


sns.heatmap(jc_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Jaccard Coefficient Heatmap")
plt.show()

sns.heatmap(smc_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("SMC Heatmap")
plt.show()

sns.heatmap(cos_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Cosine Similarity Heatmap")
plt.show()
