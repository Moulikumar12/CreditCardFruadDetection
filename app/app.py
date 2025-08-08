import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Load models
rf_model = joblib.load("artifacts/rf_model.joblib")
xgb_model = joblib.load("artifacts/xgb_model.joblib")

# Streamlit UI
st.set_page_config(page_title="Credit Card Fraud Detector", layout="centered")

st.title("💳 Credit Card Fraud Detection System")
st.write("Enter the transaction details to predict if it's fraudulent or not.")

# Create inputs for all 30 features
feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
input_data = []

col1, col2 = st.columns(2)
for idx, feature in enumerate(feature_names):
    if idx % 2 == 0:
        val = col1.number_input(f"{feature}", value=0.0)
    else:
        val = col2.number_input(f"{feature}", value=0.0)
    input_data.append(val)

# Convert to NumPy array and reshape
features_array = np.array(input_data).reshape(1, -1)

# Prediction
if st.button("Predict (Random Forest)"):
    prediction = rf_model.predict(features_array)[0]
    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

if st.button("Predict (XGBoost)"):
    prediction = xgb_model.predict(features_array)[0]
    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

# Show model metrics
st.subheader("📊 Model Performance Metrics")
if os.path.exists("artifacts/confusion_matrix_rf.png"):
    st.image("artifacts/confusion_matrix_rf.png", caption="Random Forest Confusion Matrix")
if os.path.exists("artifacts/confusion_matrix_xgb.png"):
    st.image("artifacts/confusion_matrix_xgb.png", caption="XGBoost Confusion Matrix")
if os.path.exists("artifacts/roc_curve_rf.png"):
    st.image("artifacts/roc_curve_rf.png", caption="Random Forest ROC Curve")
if os.path.exists("artifacts/roc_curve_xgb.png"):
    st.image("artifacts/roc_curve_xgb.png", caption="XGBoost ROC Curve")
