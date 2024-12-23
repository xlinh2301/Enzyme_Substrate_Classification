import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import random
from predict_and_display import predict_and_display_with_shap
from eda import eda
from preprocessing import preprocessing
# Load models đã lưu
with open("./checkpoint/xgboost_SMOTE_EC1_model.pkl", "rb") as file:
    model_ec1 = pickle.load(file)

with open("./checkpoint/xgboost_SMOTE_EC2_model.pkl", "rb") as file:
    model_ec2 = pickle.load(file)

# Load dataset
@st.cache_data
def load_data():
    dataset = pd.read_csv('./data/train.csv')
    train = dataset.drop(['id', 'EC3', 'EC4', 'EC5', 'EC6'], axis=1)
    df_new = train.copy()
    y_ec1 = dataset['EC1']
    y_ec2 = dataset['EC2']
    return df_new, y_ec1, y_ec2

df_new, y_ec1, y_ec2 = load_data()

df_test = df_new.drop(['EC1', 'EC2'], axis=1)

# Chia dữ liệu và reset index
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(df_test, y_ec1, test_size=0.20, random_state=42)
x_test_1 = x_test_1.reset_index(drop=True)
y_test_1 = y_test_1.reset_index(drop=True)

# Chia dữ liệu và reset index
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(df_test, y_ec2, test_size=0.20, random_state=42)
x_test_2 = x_test_2.reset_index(drop=True)
y_test_2 = y_test_2.reset_index(drop=True)

df_test_split = pd.concat([x_test_1, y_test_1, y_test_2], axis=1)
# print(f"x_test_1: {x_test_1}")
# print(f"y_test_1: {y_test_1}")
# print(f"y_test_2: {y_test_2}")
X_smote_ec1, y_smote_ec1, X_smote_ec2, y_smote_ec2 = preprocessing(df_test_split)
y_test_concat = pd.concat([y_smote_ec1, y_smote_ec2], axis=1)
# Sidebar để chọn trang
page = st.sidebar.selectbox("Chọn trang", ["EDA", "Dự đoán EC1", "Dự đoán EC2"])

# Hiển thị trang tương ứng
if page == "EDA":
    st.title("EDA")
    eda(df_new)
elif page == "Dự đoán EC1":
    st.title("Dự đoán EC1")
    predict_and_display_with_shap(model_ec1, X_smote_ec1, y_smote_ec1, y_test_concat, "EC1")
elif page == "Dự đoán EC2":
    st.title("Dự đoán EC2")
    predict_and_display_with_shap(model_ec2, X_smote_ec2, y_smote_ec2, y_test_concat, "EC2")
