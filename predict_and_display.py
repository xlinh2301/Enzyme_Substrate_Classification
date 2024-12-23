import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np

def format_result(value):
    """Chuyển đổi kết quả thành biểu tượng và màu sắc."""
    return ":white_check_mark: **Có**" if value == 1 else ":x: **Không**"

def display_test_data_plot(x_test_no_target):
    """Hiển thị biểu đồ dữ liệu test và cho phép chọn điểm với các nốt có màu sắc tương ứng với giá trị EC1 và EC2."""
    st.write("### Biểu đồ Dữ liệu Test")

    # Tạo cột 'case' xác định trường hợp
    def get_case(row):
        if row['EC1'] == 1 and row['EC2'] == 1:
            return '1-1'
        elif row['EC1'] == 1 and row['EC2'] == 0:
            return '1-0'
        elif row['EC1'] == 0 and row['EC2'] == 0:
            return '0-0'
        elif row['EC1'] == 0 and row['EC2'] == 1:
            return '0-1'

    x_test_no_target['case'] = x_test_no_target.apply(get_case, axis=1)

    # Ánh xạ màu sắc
    color_map = {
        '1-1': 'red',
        '1-0': 'blue',
        '0-0': 'yellow',
        '0-1': 'black',
    }

    # Biểu đồ Scatter
    fig = px.scatter(
        x_test_no_target,
        x=x_test_no_target.columns[0],
        y=x_test_no_target.columns[1],
        color='case',
        color_discrete_map=color_map,
        hover_data=x_test_no_target[['Chi1', 'BertzCT', 'EC1', 'EC2', 'case']],
        size="Chi1",
        title="Biểu đồ Scatter Dữ liệu Test"
    )

    # Hiển thị biểu đồ trong Streamlit
    selected_points = st.plotly_chart(fig, key="iris", on_select="rerun")
    print(selected_points)
    return selected_points


def display_selected_row(x_test_no_target, selection):
    """Hiển thị hàng được chọn từ điểm được chọn trên biểu đồ."""
    if selection and "selection" in selection and "points" in selection["selection"]:
        points = selection["selection"]["points"]
        
        if points:
            # Lấy tọa độ điểm đầu tiên được chọn
            clicked_point = points[0]
            clicked_x = clicked_point.get("x")
            clicked_y = clicked_point.get("y")

            # Tìm khoảng cách gần nhất
            distances = np.sqrt(
                (x_test_no_target.iloc[:, 0] - clicked_x) ** 2 +
                (x_test_no_target.iloc[:, 1] - clicked_y) ** 2
            )
            closest_point_idx = distances.idxmin()

            # Dữ liệu điểm gần nhất
            selected_data = x_test_no_target.iloc[closest_point_idx]

            st.write("### Dữ liệu của điểm được chọn:")
            st.dataframe(selected_data.to_frame().T)

            return selected_data

    st.write("Không có điểm nào được chọn.")
    return None


def make_prediction(model, selected_data, y_test, label):
    """Thực hiện dự đoán và hiển thị kết quả."""
    with st.spinner("Đang thực hiện dự đoán..."):
        selected_features = selected_data.drop(['EC1', 'EC2', 'case'])
        sample = selected_features.values.reshape(1, -1)
        prediction = model.predict(sample)

        result = format_result(prediction[0])
        row_index = selected_data.name
        ground_truth = format_result(y_test.iloc[row_index])
        st.session_state.prediction_result = result
        st.session_state.ground_truth = ground_truth
        st.session_state.sample = sample
        # st.write(f"### Kết quả dự đoán: {label} - {result}")
        # st.write(f"### Ground Truth: {label} - {ground_truth}")


def display_shap_plot(model, x_test_no_target, feature_names):
    """Hiển thị biểu đồ SHAP."""
    with st.spinner("Đang tạo biểu đồ SHAP..."):
        explainer = shap.Explainer(model, x_test_no_target, feature_names=feature_names)
        shap_values = explainer(st.session_state.sample)

        st.write("### Biểu đồ SHAP: Đóng góp của từng đặc trưng vào dự đoán")
        shap.plots.waterfall(shap_values[0])
        st.pyplot(plt)

def predict_and_display_with_shap(model, x_test, y_test, y_test_concat, label):
    st.title(f"🔍 Dự đoán {label} với SHAP")
    # Loại bỏ EC1 và EC2 trước khi dự đoán
    x_test_no_target = x_test.copy()
    feature_names = x_test_no_target.columns
    
    x_plot = pd.concat([x_test_no_target, y_test_concat], axis=1)
    # Hiển thị biểu đồ dữ liệu test
    selected_points = display_test_data_plot(x_plot)

    # Nhận hàng được chọn từ biểu đồ
    if selected_points:
        selected_data = display_selected_row(x_plot, selected_points)
        
        # Thực hiện dự đoán nếu có điểm được chọn
        if selected_data is not None and st.button("Dự đoán"):
            make_prediction(model, selected_data, y_test, label)

        if "prediction_result" in st.session_state:
            st.write(f"### Kết quả dự đoán: {label} - {st.session_state.prediction_result}")
            st.write(f"### Ground Truth: {label} - {st.session_state.ground_truth}")
        
        # Hiển thị biểu đồ SHAP nếu đã dự đoán
        if "prediction_result" in st.session_state and st.button("Hiển thị biểu đồ SHAP"):
            display_shap_plot(model, x_test_no_target, feature_names)

