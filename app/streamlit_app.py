import streamlit as st
import pandas as pd
import requests
import os

st.set_page_config(page_title="Credit Card Fraud Demo", layout="wide")
st.title("Credit Card Fraud Detection — Demo UI")

# Use environment variable API_URL if set, else default to your deployed URL
default_api_url = os.getenv("API_URL", "https://creditcardfruaddetection-2.onrender.com/predict")
api_url = st.text_input("API URL", default_api_url)

st.markdown("**Instructions:** Provide exactly 30 features in order: `Time, V1..V28, Amount`.")

st.sidebar.header("Input mode")
mode = st.sidebar.radio("Mode", ["Single (paste)", "Single (form)", "Upload CSV"])

def call_api(payload, url, timeout=20):
    try:
        with st.spinner("Predicting..."):
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()  # Raise HTTPError for bad responses
            return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
    return None

if mode == "Single (paste)":
    sample_text = st.text_area("Paste 30 comma-separated values (Time, V1..V28, Amount)", height=120)
    if st.button("Predict (paste)"):
        values = []
        try:
            values = [float(x.strip()) for x in sample_text.split(",") if x.strip() != ""]
        except ValueError:
            st.error("Please ensure all inputs are valid numbers.")
        if len(values) != 30:
            st.error(f"Need exactly 30 values, got {len(values)}")
        else:
            result = call_api({"features": values}, api_url)
            if result:
                st.json(result)

elif mode == "Single (form)":
    cols = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]
    st.markdown("Enter values (defaults 0) — easier for quick test")
    values = []
    for c in cols:
        v = st.number_input(c, value=0.0, format="%.6f")
        values.append(float(v))
    if st.button("Predict (form)"):
        result = call_api({"features": values}, api_url)
        if result:
            st.json(result)

else:  # Upload CSV
    uploaded = st.file_uploader("Upload CSV with columns (Time,V1..V28,Amount) or same order", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())
        expected = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]
        if all(col in df.columns for col in expected):
            df_features = df[expected]
        else:
            st.warning("CSV does not exactly contain named columns. Using first 30 columns as features.")
            df_features = df.iloc[:, :30]

        if st.button("Predict CSV (batch)"):
            instances = df_features.values.tolist()
            payload = {"instances": instances}
            result = call_api(payload, api_url, timeout=60)
            if result:
                if "results" in result:
                    st.table(pd.DataFrame(result["results"]))
                else:
                    st.json(result)
