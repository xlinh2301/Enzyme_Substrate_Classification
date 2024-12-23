# Streamlit App for EC1 and EC2 Prediction

This Streamlit app provides an interface for exploratory data analysis (EDA) and prediction of EC1 and EC2 values using machine learning models. The app allows users to explore the dataset, visualize the distribution of features, and make predictions using pre-trained XGBoost models. It also supports SHAP analysis for model interpretability.

## Datasets

You can obtain the dataset from Kaggle: [Playground Series S3E18 - Enzyme Substrate Classification](https://www.kaggle.com/competitions/playground-series-s3e18).

## Features

1. **Exploratory Data Analysis (EDA)**:
   - **Univariate analysis**: Visualize histograms and boxplots of individual features.
   - **Bivariate analysis**: Visualize violin plots for features with target variables (EC1, EC2).
   - **Multivariate analysis**: Create scatter plots for multiple feature comparisons.

2. **Prediction**:
   - Make predictions for EC1 and EC2 using pre-trained XGBoost models.
   - **SHAP (SHapley Additive exPlanations)** analysis for model interpretability and feature importance.

3. **Data Preprocessing**:
   - Remove outliers from the dataset using the interquartile range (IQR).
   - Normalize data for improved model performance.
   - Apply **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset.
   
## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/xlinh2301/Enzyme_Substrate_Classification
   cd Enzyme_Substrate_Classification
   ```
2. Install necessary library:

  ```bash
  pip install requirements.txt
  ```
## Folder structure
```bash
FINAL_PROJECT/
   │
   ├── data/                   # Contains datasets
   │   ├── train.csv           # Original dataset
   │
   ├── checkpoint/                 # Pre-trained models
   │   ├── xgboost_model_ec1.pkl
   │   ├── xgboost_model_ec2.pkl
   │   ├── xgboost_SMOTE_EC1_model.pkl
   │   ├── xgboost_SMOTE_EC2_model.pkl
   │
   ├── results/                # Results and visualizations
   │   ├── violin_plots_EC1.png
   │   ├── violin_plots_EC2.png
   │
   ├── src/                    # Source code
   │   ├── app.py              # Main entry point for the Streamlit app
   │   ├── preprocessing.py    # Functions for data preprocessing
   │   ├── predict_and_display.py  # Functions for prediction and SHAP analysis
   │
   ├── .gitignore              # To exclude unnecessary files from version control
   ├── README.md               # Project documentation
   ├── requirements.txt        # List of required Python libraries
   └── LICENSE                 # License file (optional)
```
## Files
  - ``app.py``: Main entry point for the Streamlit app.
  - ``eda.py``: Contains functions for exploratory data analysis (EDA).
  - ``predict_and_display.py``: Includes the prediction functions and SHAP analysis.
  - ``train.csv``: Dataset containing the features and target variables (EC1, EC2).
  - ``xgboost_SMOTE_EC1_model``: Pre-trained XGBoost model for EC1 prediction.
  - ``xgboost_SMOTE_EC1_model``: Pre-trained XGBoost model for EC2 prediction.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
  - Nguyen Xuan Linh
