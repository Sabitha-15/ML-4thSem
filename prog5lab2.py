import numpy as np
import pandas as pd

def load_data(file_path):
    return pd.read_excel(
        file_path,
        sheet_name=2,
        usecols='B:AE'
         )

def prepare_classify_data(data_frame):
    data_frame.replace('?',pd.NA,inplace=True)
    numeric_like_cols = ['TSH','T3','TT4','T4U','FTI','TBG']

    for col in numeric_like_cols:
        data_frame[col] = pd.to_numeric(data_frame[col], errors='coerce')

    numerical=data_frame.select_dtypes(include=['int64','float64','int32','float32']).columns.tolist()
    
    categorical=data_frame.select_dtypes(include='object').columns.tolist()
    
    binary=[cat for cat in categorical if data_frame[cat].dropna().isin(['t','f']).all()]
    for cat in binary:
        if cat in categorical:
            categorical.remove(cat)
    return (numerical, categorical,binary)

def jaccard_SMC(df):
    row1=df.iloc[0,:]
    row2=df.iloc[1,:]
    f00=f10=f11=f01=0

    for i,j in zip(row1,row2):
            if i==1 and j==1:
                f11+=1
            elif i==1 and j==0:
                f10+=1
            elif i==0 and j==1:
                f01+=1
            else:
                f00+=1

    if (f10 + f01 + f11) == 0:
     jaccard_coefficient = 0  # or set to None
    else:
     jaccard_coefficient = f11 / (f10 + f01 + f11)

    simple_matching_coefficient=(f00+f11)/(f00+f01+f10+f11)
    
    return jaccard_coefficient,simple_matching_coefficient
    
def main():
    file_path='Lab Session Data (1).xlsx'
    df=load_data(file_path)
    x,y,z=prepare_classify_data(df)
    df1 = df.loc[0:1, z].replace({'t':1,'f':0})
    df1 = df1.astype(int)
    jac,smc=jaccard_SMC(df1)
    print(jac)
    print(smc)    
    
main()