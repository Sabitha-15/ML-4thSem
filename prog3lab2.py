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

def data_range(df,col):
    data=df[col].to_numpy()
    min_data=np.nanmin(data)
    max_data=np.nanmax(data)
    
    return (min_data,max_data)

def mean_varaince_standard_dev(df,col):
    data=df[col].to_numpy()
    mean_data=np.nanmean(data)
    var_data=np.nanvar(data)
    standard_deviation=var_data**0.5
    
    return mean_data,var_data,standard_deviation

def finding_outliers(df,col):
    data = df[col].dropna() 
    q1=data.quantile(0.25)
    q3=data.quantile(0.75)
    iqr=q3-q1
    lower_bound=q1-1.5*iqr
    upper_bound=q3+1.5*iqr
    outliers=data[(data<lower_bound) | (data>upper_bound)]
    
    return outliers
    

def main():
    file_path='Lab Session Data (1).xlsx'
    df=load_data(file_path)
    x,y,z=prepare_classify_data(df) 
    print(x)
    print('\n',y,'\n')
    print('\n',z,'\n')
    for col in x:
        min_val, max_val = data_range(df, col)
        mean_val, var_val, std_val = mean_varaince_standard_dev(df, col)
        outliers = finding_outliers(df, col)
        
        print(f"\nColumn: {col}")
        print(f"Min: {min_val}, Max: {max_val}")
        print(f"Mean: {mean_val:.3f}, Variance: {var_val:.3f}, Std Dev: {std_val:.3f}")
        print(f"Outliers: ",outliers)
    
main()