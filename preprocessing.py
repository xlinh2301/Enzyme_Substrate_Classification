import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import json
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
from imblearn.over_sampling import SMOTE

def replace_outliers_with_bounds(df):
    df_adjusted = df.copy()
    outliers_count = 0  # Biến đếm số lượng outlier được xử lý
    
    for column in df_adjusted.select_dtypes(include=['float64', 'int64']):
        Q1 = df_adjusted[column].quantile(0.25)  
        Q3 = df_adjusted[column].quantile(0.75)  
        IQR = Q3 - Q1  # Tính khoảng IQR
        
        # Tính khoảng cho phép
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Đếm số lượng outlier trong cột hiện tại
        column_outliers = ((df_adjusted[column] < lower_bound) | (df_adjusted[column] > upper_bound)).sum()
        outliers_count += column_outliers  # Cộng vào tổng số outlier
        
        # Thay thế outlier
        df_adjusted[column] = df_adjusted[column].apply(
            lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x)
        )
    
    print(f"Số lượng outlier đã xử lý: {outliers_count}")
    return df_adjusted

def apply_log_transform_positive(X):
    X_transformed = X.copy()
    X_transformed = X_transformed.applymap(lambda x: np.log1p(x) if x > 0 else 0)
    return pd.DataFrame(X_transformed, columns=X.columns)

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote

def preprocessing(df):
    label_columns = ['EC1', 'EC2']
    labels = df[['EC1', 'EC2']].copy()
    df_temp = df
    df_temp.drop(['EC1','EC2'], axis = 1, inplace = True)
    df_adjusted_with_IQR = pd.concat([replace_outliers_with_bounds(df_temp), labels], axis=1)
    df_main = df_adjusted_with_IQR.copy()
    log_transformed_data = apply_log_transform_positive(df_main.drop(columns=label_columns))
    log_transformed_df = pd.concat([log_transformed_data, df_main[label_columns]], axis=1)
    X = log_transformed_df.drop(columns=label_columns)
    y_ec1 = log_transformed_df['EC1']
    y_ec2 = log_transformed_df['EC2']
    # print(f"df: {df}")
    # print(f"df_adjusted_with_IQR: {df_adjusted_with_IQR}")
    # print(f"log_transformed_df: {log_transformed_df}")

    # print(f"y_ec1: {y_ec1}")
    X_smote_ec1, y_smote_ec1 = apply_smote(X, y_ec1)
    X_smote_ec2, y_smote_ec2 = apply_smote(X, y_ec2)
    return X_smote_ec1, y_smote_ec1, X_smote_ec2, y_smote_ec2