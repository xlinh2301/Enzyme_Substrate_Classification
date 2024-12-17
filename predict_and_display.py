import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import random

def format_result(value):
    """Chuyển đổi kết quả thành biểu tượng và màu sắc."""
    return ":white_check_mark: **Có**" if value == 1 else ":x: **Không**"

def display_test_data(x_test_no_target):
    """Hiển thị bảng dữ liệu test."""
    st.write("### Dữ liệu Test")
    st.dataframe(x_test_no_target)

def get_user_input(x_test_no_target):
    """Nhận số thứ tự của hàng từ người dùng hoặc chọn ngẫu nhiên với nút Load lại."""
    st.write("### Chọn hàng để dự đoán")
    choice = st.radio("Chọn cách lựa chọn hàng", ["Chọn hàng", "Chọn ngẫu nhiên"])

    if choice == "Chọn hàng":
        row_number = st.number_input(
            f"Nhập số từ 0 đến {len(x_test_no_target) - 1}", 
            min_value=0, max_value=len(x_test_no_target) - 1, step=1
        )
        return row_number
    
    elif choice == "Chọn ngẫu nhiên":
        # Tạo một nút để load lại số ngẫu nhiên
        if 'random_row' not in st.session_state:
            st.session_state.random_row = random.randint(0, len(x_test_no_target) - 1)

        load_button = st.button("Load lại số ngẫu nhiên")

        if load_button:
            st.session_state.random_row = random.randint(0, len(x_test_no_target) - 1)
            st.write(f"Số ngẫu nhiên mới đã được chọn: {st.session_state.random_row}")
        else:
            st.write(f"Hàng được chọn ngẫu nhiên: {st.session_state.random_row}")
        return st.session_state.random_row

def display_selected_row(selected_row):
    """Hiển thị hàng được chọn."""
    st.write("### Hàng được chọn:")
    st.dataframe(selected_row.to_frame().T)

def make_prediction(model, selected_row, y_test, row_number, label):
    """Thực hiện dự đoán và hiển thị kết quả."""
    with st.spinner("Đang thực hiện dự đoán..."):
        sample = selected_row.values.reshape(1, -1)
        prediction = model.predict(sample)

        result = format_result(prediction[0])
        ground_truth = format_result(y_test.iloc[row_number])

        # Lưu kết quả vào session_state
        st.session_state.prediction_result = result
        st.session_state.ground_truth = ground_truth
        st.session_state.sample = sample

def display_shap_plot(model, x_test_no_target, feature_names):
    """Hiển thị biểu đồ SHAP."""
    with st.spinner("Đang tạo biểu đồ SHAP..."):
        explainer = shap.Explainer(model, x_test_no_target, feature_names=feature_names)
        shap_values = explainer(st.session_state.sample)

        st.write("### Biểu đồ SHAP: Đóng góp của từng đặc trưng vào dự đoán")
        shap.plots.waterfall(shap_values[0])
        st.pyplot(plt)

def predict_and_display_with_shap(model, x_test, y_test, label):
    st.title(f"🔍 Dự đoán {label} với SHAP")

    # Loại bỏ EC1 và EC2 trước khi dự đoán
    x_test_no_target = x_test.drop(['EC1', 'EC2'], axis=1)
    feature_names = x_test_no_target.columns

    # Hiển thị dữ liệu test
    display_test_data(x_test_no_target)

    # Nhận hàng được chọn từ người dùng
    row_number = get_user_input(x_test_no_target)
    selected_row = x_test_no_target.iloc[row_number]

    # Hiển thị hàng được chọn
    display_selected_row(selected_row)

    # Thực hiện dự đoán
    if st.button("Dự đoán"):
        make_prediction(model, selected_row, y_test, row_number, label)

    if "prediction_result" in st.session_state:
        st.write(f"### Kết quả dự đoán: {label} - {st.session_state.prediction_result}")
        st.write(f"### Ground Truth: {label} - {st.session_state.ground_truth}")

    # Hiển thị biểu đồ SHAP nếu đã dự đoán
    if "prediction_result" in st.session_state and st.button("Hiển thị biểu đồ SHAP"):
        display_shap_plot(model, x_test_no_target, feature_names)