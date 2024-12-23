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
                
    with tabs[1]:
        st.header("📈 Phân Tích Hai Biến")
        subtabs = st.tabs(["Correlation with EC1",  "Correlation with EC2"])
        # # Chọn biến mục tiêu (ví dụ: EC1)
        # target = st.selectbox("Chọn biến mục tiêu", ["EC1", "EC2"])
        
        # # Đường dẫn lưu trữ ảnh
        # image_path = f"violin_plots_{target}.png"

        # st.subheader(f"Violin Plot cho từng đặc trưng với {target}")
        # st.image(image_path, caption="Violin Plots", use_container_width=True)
        
         # Tính toán ma trận tương quan
        corr = df.corr()

        # Lấy tương quan mà không bao gồm 'EC1' và 'EC2'
        ec1_corr = corr['EC1'].drop(['EC1', 'EC2'])
        ec2_corr = corr['EC2'].drop(['EC1', 'EC2'])

        # Sắp xếp giá trị tương quan giảm dần
        ec1_corr_sorted = ec1_corr.sort_values(ascending=False)
        ec2_corr_sorted = ec2_corr.sort_values(ascending=False)

        with subtabs[0]:
            # Hiển thị heatmap tương quan với EC1
            st.subheader("Heatmap tương quan với EC1")
            sns.set_style("white")
            sns.set_palette("PuBuGn_d")
            fig1, ax1 = plt.subplots()
            sns.heatmap(ec1_corr_sorted.to_frame(), cmap="coolwarm", annot=True, fmt='.2f', ax=ax1)
            ax1.set_title('Correlation with EC1')
            st.pyplot(fig1)

        with subtabs[1]:
            # Hiển thị heatmap tương quan với EC2
            st.subheader("Heatmap tương quan với EC2")
            fig2, ax2 = plt.subplots()
            sns.heatmap(ec2_corr_sorted.to_frame(), cmap="coolwarm", annot=True, fmt='.2f', ax=ax2)
            ax2.set_title('Correlation with EC2')
            st.pyplot(fig2)

    with tabs[2]:
        st.header("🔍 Phân Tích Đa Biến")
        st.write('---')
        
        st.subheader("Correlation Matrix")
        # Calculate the correlation matrix
        corr = df.corr()

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
        st.pyplot(f)
    with tabs[3]:
        # --- Tabs Layout ---
        tabs = st.tabs(["Chọn Feature", "Loại Bỏ Outliers", "Scale Data", "SMOTE"])
        corr = df.corr()

        union_cols_df = df.copy()
        union_cols_df = union_cols_df.drop(['EC1', 'EC2'], axis=1)
        labels = df[['EC1', 'EC2']]
        # --- Tab 0: Chọn Feature ---
        with tabs[0]:
            st.subheader("Chọn Feature để xử lý")
 
            selected_feature = st.selectbox("Chọn đặc trưng để xử lý", union_cols_df.columns, key="selected_feature")
            
            st.write(f"Feature được chọn: **{selected_feature}**")

        # --- Tab 1: Loại Bỏ Outliers ---
        with tabs[1]:
            st.subheader("Thay Thế Outliers")

            # Kiểm tra session state để lấy feature đã chọn
            if "selected_feature" not in st.session_state:
                st.warning("Hãy chọn một feature ở tab 'Chọn Feature' trước khi tiếp tục.")
            else:
                selected_feature = st.session_state.selected_feature
                union_cols_df_to_use = st.session_state.get("union_cols_df_filtered", union_cols_df)
            
                if st.button("Thay thế outliers"):
                    with st.spinner("Đang xử lý... vui lòng chờ."):
                        # Tính toán các giá trị Q1, Q3 và IQR
                        q1 = union_cols_df_to_use[selected_feature].quantile(0.25)
                        q3 = union_cols_df_to_use[selected_feature].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outliers_count = 0
                        column_outliers = ((union_cols_df_to_use[selected_feature] < lower_bound) | (union_cols_df_to_use[selected_feature] > upper_bound)).sum()
                        outliers_count += column_outliers  # Cộng vào tổng số outlier

                        # Thay thế outliers bằng giá trị giới hạn
                        union_cols_df_to_use[selected_feature] = union_cols_df_to_use[selected_feature].apply(
                            lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x)
                        )
                        # Lưu lại DataFrame sau khi xử lý
                        st.session_state.union_cols_df_filtered = union_cols_df_to_use
                        print(f"union_cols_df_filtered: {union_cols_df_to_use}")
                        st.write(f"Số lượng outlier đã xử lý: {outliers_count}")

                        # Biểu đồ
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Trước khi xử lý outliers**")
                            fig_before = px.box(df, y=selected_feature)
                            st.plotly_chart(fig_before, key="fig_before")
                        with col2:
                            union_cols_df_to_use = pd.concat([union_cols_df_to_use, labels], axis=1)
                            st.write("**Sau khi xử lý outliers**")
                            fig_after = px.box(union_cols_df_to_use, y=selected_feature)
                            st.plotly_chart(fig_after, key="fig_after")

        # --- Tab 2: Scale Data ---
        with tabs[2]:
            st.subheader("Scale Data")

            # Kiểm tra session state để lấy dữ liệu đã loại bỏ outliers
            if "union_cols_df_filtered" not in st.session_state:
                st.warning("Hãy thực hiện bước 'Loại Bỏ Outliers' trước khi tiếp tục.")
            else:
                filtered_df = st.session_state.union_cols_df_filtered

                if st.button("Scale"):
                    with st.spinner("Đang scale... vui lòng chờ."):
                        def apply_log_transform_positive(X):
                            X_transformed = X.copy()
                            X_transformed = X_transformed.applymap(lambda x: np.log1p(x) if x > 0 else 0)
                            return pd.DataFrame(X_transformed, columns=X.columns)
                        normalized_df = apply_log_transform_positive(filtered_df)
                        st.session_state.union_cols_df_normalized = normalized_df  # Lưu lại dữ liệu đã normalize

                        st.write("Dữ liệu sau khi scale:")
                        st.write(normalized_df.head())

                        # Biểu đồ
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Trước khi scale**")
                            fig_before_norm = px.histogram(filtered_df, x=selected_feature)
                            st.plotly_chart(fig_before_norm, key="before_norm")
                        with col2:
                            st.write("**Sau khi scale**")
                            fig_after_norm = px.histogram(normalized_df, x=selected_feature)
                            st.plotly_chart(fig_after_norm, key="after_norm")

        # --- Tab 3: SMOTE ---
        with tabs[3]:
            st.subheader("Cân Bằng Dữ Liệu với SMOTE")

            # Kiểm tra session state để lấy dữ liệu đã normalize
            if "union_cols_df_normalized" not in st.session_state:
                st.warning("Hãy thực hiện bước 'Normalize Data' trước khi tiếp tục.")
            else:
                normalized_df = st.session_state.union_cols_df_normalized
                print(f"normalized_df: {normalized_df}")
                target_col = st.selectbox("Chọn cột target", ['EC1', 'EC2'], key="target_column")

                if st.button("Áp dụng SMOTE"):
                    with st.spinner("Đang cân bằng dữ liệu... vui lòng chờ."):
                        X = normalized_df[[selected_feature]].dropna()
                        y = labels[target_col]
                        # print(f"X: {X}")
                        # print(f"y: {y}")

                        # Áp dụng SMOTE
                        smote = SMOTE(random_state=42)
                        X_resampled, y_resampled = smote.fit_resample(X, y)

                        # Tạo DataFrame mới sau khi SMOTE
                        smote_df = pd.DataFrame(X_resampled, columns=[selected_feature])
                        smote_df[target_col] = y_resampled
                        st.session_state.union_cols_df_smote = smote_df  # Lưu lại dữ liệu đã cân bằng

                        st.write(f"Số lượng mẫu trước khi SMOTE: {len(X)}")
                        st.write(f"Số lượng mẫu sau khi SMOTE: {len(smote_df)}")

                        df_before_smote = pd.concat([normalized_df[[selected_feature]], labels], axis=1)
                        df_after_smote = pd.concat([smote_df[[selected_feature]], smote_df[target_col]], axis=1)
                        col1, col2 = st.columns(2)
                        with col1:
                            # Biểu đồ trước SMOTE
                            st.write("**Trước khi SMOTE:**")
                            fig_before_smote = px.histogram(df_before_smote, x=selected_feature, color=target_col, title="Trước SMOTE")
                            st.plotly_chart(fig_before_smote)

                        with col2:
                            # Biểu đồ sau SMOTE
                            st.write("**Sau khi SMOTE:**")
                            
                            fig_after_smote = px.histogram(df_after_smote, x=selected_feature, color=target_col, title="Sau SMOTE")
                            st.plotly_chart(fig_after_smote)
                        col3, col4 = st.columns(2)

                        with col3:
                            # Biểu đồ countplot trước SMOTE
                            st.write("**Trước khi SMOTE:**")
                            plt.figure(figsize=(8, 6))
                            sns.countplot(x=target_col, data=df_before_smote)
                            st.pyplot(plt)

                        with col4:
                            # Biểu đồ countplot sau SMOTE
                            st.write("**Sau khi SMOTE:**")
                            plt.figure(figsize=(8, 6))
                            sns.countplot(x=target_col, data=df_after_smote)
                            st.pyplot(plt)
