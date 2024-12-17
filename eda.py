import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import json
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
from imblearn.over_sampling import SMOTE

SAVE_PATH = "results"  # Directory to store results
EC1_FEATURES_FILE = os.path.join(SAVE_PATH, "selected_features_ec1.json")
EC2_FEATURES_FILE = os.path.join(SAVE_PATH, "selected_features_ec2.json")
EC1_DF_FILE = os.path.join(SAVE_PATH, "df_selected_ec1.pkl")
EC2_DF_FILE = os.path.join(SAVE_PATH, "df_selected_ec2.pkl")
os.makedirs(SAVE_PATH, exist_ok=True)

if 'df_filtered' not in st.session_state:
    st.session_state.df_filtered = None

def eda(df):
    # Tạo các tab chính
    tabs = st.tabs(["Phân Tích Đơn Biến", "Phân Tích Hai Biến", "Phân Tích Đa Biến", "Xử Lý Dữ Liệu"])

    with tabs[0]:
        st.header("📊 Phân Tích Đơn Biến")

        # Chọn đặc trưng để phân tích
        selected_feature = st.selectbox("Chọn đặc trưng để phân tích", df.columns)

        if selected_feature:
            # Tạo 2 cột
            col1, col2 = st.columns(2)

            # Histogram trong cột 1
            with col1:
                st.subheader("Histogram")
                fig, ax = plt.subplots()
                sns.histplot(df[selected_feature], kde=True, ax=ax)
                st.pyplot(fig)

            # Boxplot trong cột 2
            with col2:
                st.subheader("Boxplot")
                fig, ax = plt.subplots()
                sns.boxplot(y=df[selected_feature], ax=ax)
                st.pyplot(fig)

            # # Bar Plot (cho dữ liệu phân loại)
            # st.subheader("Bar Plot")
            # if df[selected_feature].dtype == 'object':
            #     fig, ax = plt.subplots()
            #     sns.countplot(x=df[selected_feature], ax=ax)
            #     st.pyplot(fig)
            # else:
            #     st.write("Không thể tạo Bar Plot cho dữ liệu số.")
    with tabs[1]:
        st.header("📈 Phân Tích Hai Biến")
        
        # Chọn biến mục tiêu (ví dụ: EC1)
        target = st.selectbox("Chọn biến mục tiêu", ["EC1", "EC2"])
        
        # Đường dẫn lưu trữ ảnh
        image_path = f"violin_plots_{target}.png"

        st.subheader(f"Violin Plot cho từng đặc trưng với {target}")
        st.image(image_path, caption="Violin Plots", use_container_width=True)
        

    with tabs[2]:
        st.header("🔍 Phân Tích Đa Biến")

        # Separate the features (X) and target variables (y)
        X_ec1 = df.drop(['EC2'], axis=1)
        X_ec2 = df.drop(['EC1'], axis=1)
        
        # Scatter Plot for Two Features
        tab_ec1, tab_ec2 = st.tabs(["Scatter Plot - EC1", "Scatter Plot - EC2"])
        with tab_ec1:
            st.subheader("Scatter Plot - EC1")
            # Select the two features   for the scatter plot - EC1
            feature_x_ec1 = st.selectbox("Chọn đặc trưng X - EC1", X_ec1.columns, key="feature_x_ec1")
            feature_y_ec1 = st.selectbox("Chọn đặc trưng Y - EC1", X_ec1.columns, key="feature_y_ec1")

            with st.spinner("Đang tạo Scatter Plot EC1..."):
                # Create the scatter plot using Plotly
                fig_scatter = px.scatter(X_ec1, 
                                        x=feature_x_ec1, 
                                        y=feature_y_ec1, 
                                        color='EC1', 
                                        title=f'Scatter Plot - {feature_x_ec1} vs {feature_y_ec1} - EC1',
                                        color_continuous_scale='Viridis')  # Choose a valid predefined colorscale

                # Display the plot in Streamlit
                st.plotly_chart(fig_scatter)
            st.write("---")
        with tab_ec2:
            st.subheader("Scatter Plot - EC2")
            # Select the two features for the scatter plot - EC2
            feature_x_ec2 = st.selectbox("Chọn đặc trưng X - EC2", X_ec2.columns, key="feature_x_ec2")
            feature_y_ec2 = st.selectbox("Chọn đặc trưng Y - EC2", X_ec2.columns, key="feature_y_ec2")

            with st.spinner("Đang tạo Scatter Plot EC2..."):
                # Create the scatter plot using Plotly
                fig_scatter_EC2 = px.scatter(X_ec2, 
                                            x=feature_x_ec2, 
                                            y=feature_y_ec2, 
                                            color='EC2', 
                                            title=f'Scatter Plot - {feature_x_ec2} vs {feature_y_ec2} - EC2',
                                            color_continuous_scale='Viridis')  # Choose a valid predefined colorscale

                # Display the plot in Streamlit
                st.plotly_chart(fig_scatter_EC2)
    with tabs[3]:
        # --- Tabs Layout ---
        tabs = st.tabs(["Loại Bỏ Outliers", "SMOTE", "Normalize Data"])
        corr = df.corr()

        # Get correlations without 'EC1' and 'EC2'
        ec1_corr = corr['EC1'].drop(['EC1', 'EC2'])
        ec1_corr_sorted = ec1_corr.sort_values(ascending=False)
        # Filter out features with correlation equal to 0.0
        features_to_keep1 = ec1_corr[abs(ec1_corr) >= 0.01].index

        # Get correlations without 'EC1' and 'EC2'
        ec2_corr = corr['EC2'].drop(['EC1', 'EC2'])
        ec2_corr_sorted = ec2_corr.sort_values(ascending=False)
        # Filter out features with correlation equal to 0.0
        features_to_keep2 = ec2_corr[abs(ec2_corr) >= 0.01].index

        union_cols = list(set(features_to_keep1) | set(features_to_keep2))

        # Print the result
        print(union_cols)
        union_cols_df = pd.DataFrame(df[union_cols])
        union_cols_df[['EC1', 'EC2']] = df[['EC1', 'EC2']]

        # --- Tab 1: Loại Bỏ Outliers ---
        with tabs[0]:
            st.subheader("Loại Bỏ Outliers")
            selected_feature = st.selectbox("Chọn đặc trưng để xử lý", union_cols_df.columns, key="outlier_feature")

            if st.button("Loại bỏ outliers"):
                with st.spinner("Đang load... vui lòng chờ."):
                    q1 = union_cols_df[selected_feature].quantile(0.25)
                    q3 = union_cols_df[selected_feature].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    # Lọc dữ liệu không chứa outliers
                    st.session_state.union_cols_df_filtered = union_cols_df[(union_cols_df[selected_feature] >= lower_bound) & (union_cols_df[selected_feature] <= upper_bound)]
                    st.write(f"Đã loại bỏ outliers, số dòng còn lại: {len(st.session_state.union_cols_df_filtered)}")

                    # Biểu đồ trước và sau khi loại bỏ outliers
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Trước khi loại bỏ outliers**")
                        fig_before = px.box(union_cols_df, y=selected_feature)
                        st.plotly_chart(fig_before)
                    with col2:
                        st.write("**Sau khi loại bỏ outliers**")
                        fig_after = px.box(st.session_state.union_cols_df_filtered, y=selected_feature)
                        st.plotly_chart(fig_after)

        # --- Tab 2: SMOTE ---
        with tabs[1]:
            st.subheader("Cân Bằng Dữ Liệu với SMOTE")
            target_col = st.selectbox("Chọn cột target", ['EC1', 'EC2'], key="target_column")

            if st.button("Áp dụng SMOTE"):
                with st.spinner("Đang cân bằng dữ liệu... vui lòng chờ."):
                    union_cols_df_smote_input = st.session_state.get("union_cols_df_filtered", union_cols_df).dropna()
                    
                    if target_col == 'EC1':
                        X = union_cols_df_smote_input.drop(columns=['EC2', 'EC1'])
                        y = union_cols_df_smote_input['EC1']
                    else:
                        X = union_cols_df_smote_input.drop(columns=['EC1', 'EC2'])
                        y = union_cols_df_smote_input['EC2']

                    # Áp dụng SMOTE
                    smote = SMOTE(random_state=42)
                    X_resampled, y_resampled = smote.fit_resample(X, y)
                    
                    # Tạo DataFrame mới sau khi SMOTE
                    union_cols_df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
                    union_cols_df_resampled[target_col] = y_resampled
                    st.session_state.union_cols_df_smote = union_cols_df_resampled

                    st.write(f"Số lượng mẫu trước khi SMOTE: {len(union_cols_df_smote_input)}")
                    st.write(f"Số lượng mẫu sau khi SMOTE: {len(union_cols_df_resampled)}")

                    # Hiển thị biểu đồ trước và sau khi SMOTE
                    col1, col2 = st.columns(2)
                    with col1:
                        if len(X.columns) >= 2:
                            fig_before_smote = px.scatter(union_cols_df_smote_input, x=X.columns[0], y=X.columns[1], color=target_col, title="Trước SMOTE")
                            st.plotly_chart(fig_before_smote)
                    with col2:
                        if len(X_resampled.columns) >= 2:
                            fig_after_smote = px.scatter(union_cols_df_resampled, x=X_resampled.columns[0], y=X_resampled.columns[1], color=target_col, title="Sau SMOTE")
                            st.plotly_chart(fig_after_smote)

        # --- Tab 3: Normalize Data ---
        with tabs[2]:
            st.subheader("Normalize Data")
            selected_feature_norm = st.selectbox("Chọn đặc trưng để normalize", union_cols_df.columns, key="normalize_feature")

            if st.button("Normalize"):
                with st.spinner("Đang normalize... vui lòng chờ."):
                    data_to_normalize = st.session_state.get("union_cols_df_smote", union_cols_df)
                    
                    # Normalize dữ liệu
                    scaler = MinMaxScaler()
                    union_cols_df_normalized = data_to_normalize.copy()
                    union_cols_df_normalized[selected_feature_norm] = scaler.fit_transform(data_to_normalize[[selected_feature_norm]])
                    
                    st.write("Dữ liệu sau khi normalize:")
                    st.write(union_cols_df_normalized.head())
                    
                    # Biểu đồ trước và sau khi normalize
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Trước khi normalize**")
                        fig_before_norm = px.histogram(data_to_normalize, x=selected_feature_norm)
                        st.plotly_chart(fig_before_norm)
                    with col2:
                        st.write("**Sau khi normalize**")
                        fig_after_norm = px.histogram(union_cols_df_normalized, x=selected_feature_norm)
                        st.plotly_chart(fig_after_norm)

