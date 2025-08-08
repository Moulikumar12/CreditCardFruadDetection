# app/streamlit_app.py
import streamlit as st
import pandas as pd
import requests
import json
import numpy as np

st.set_page_config(page_title="Credit Card Fraud Demo", layout="wide")
st.title("Credit Card Fraud Detection — Demo UI")

api_url = st.text_input("API URL", "http://127.0.0.1:5000/predict")

st.markdown("**Instructions:** Provide exactly 30 features in order: `Time, V1..V28, Amount`.")

st.sidebar.header("Input mode")
mode = st.sidebar.radio("Mode", ["Single (paste)", "Single (form)", "Upload CSV"])

if mode == "Single (paste)":
    sample_text = st.text_area("Paste 30 comma-separated values (Time, V1..V28, Amount)", height=120)
    if st.button("Predict (paste)"):
        try:
            values = [float(x.strip()) for x in sample_text.split(",") if x.strip() != ""]
            if len(values) != 30:
                st.error(f"Need 30 values, got {len(values)}")
            else:
                payload = {"features": values}
                resp = requests.post(api_url, json=payload, timeout=20)
                st.json(resp.json())
        except Exception as e:
            st.error(str(e))

elif mode == "Single (form)":
    cols = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]
    st.markdown("Enter values (defaults 0) — easier for quick test")
    values = []
    for c in cols:
        v = st.number_input(c, value=0.0, format="%.6f")
        values.append(float(v))
    if st.button("Predict (form)"):
        try:
            payload = {"features": values}
            resp = requests.post(api_url, json=payload, timeout=20)
            st.json(resp.json())
        except Exception as e:
            st.error(str(e))

else:  # Upload CSV
    uploaded = st.file_uploader("Upload CSV with columns (Time,V1..V28,Amount) or same order", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())
        # Try to reorder if names present
        expected = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]
        if all(col in df.columns for col in expected):
            df_features = df[expected]
        else:
            st.warning("CSV does not exactly contain named columns. Using first 30 columns as features.")
            df_features = df.iloc[:, :30]
        if st.button("Predict CSV (batch)"):
            instances = df_features.values.tolist()
            payload = {"instances": instances}
            try:
                resp = requests.post(api_url, json=payload, timeout=60)
                st.write(resp.json())
                # If batch result is returned as {"results": [...]} show table
                res = resp.json()
                if "results" in res:
                    st.table(pd.DataFrame(res["results"]))
                else:
                    st.json(res)
            except Exception as e:
                st.error(str(e))
