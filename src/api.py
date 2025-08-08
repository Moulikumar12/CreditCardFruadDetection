# src/api.py
from flask import Flask, request, jsonify
import joblib, os
import numpy as np

app = Flask(__name__)

MODEL_PATH = 'artifacts/xgb_model.joblib'
SCALER_PATH = 'artifacts/scaler.joblib'

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Run src/train.py first.")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

def preprocess_sample(arr):
    arr = np.array(arr, dtype=float).flatten()
    if arr.shape[0] != 30:
        raise ValueError(f"Feature shape mismatch, expected 30, got {arr.shape[0]}")
    # Scale Time (index 0) and Amount (index 29)
    if scaler is not None:
        ta = np.array([[arr[0], arr[29]]])
        ta_scaled = scaler.transform(ta)[0]
        arr[0], arr[29] = ta_scaled[0], ta_scaled[1]
    return arr.reshape(1, -1)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message":"Credit Card Fraud Detection API running."})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
    except:
        return jsonify({"error":"Invalid JSON"}), 400

    if not data:
        return jsonify({"error":"Empty JSON"}), 400

    # Support single sample or batch
    if "features" in data:
        samples = [data["features"]]
    elif "instances" in data:
        samples = data["instances"]
    else:
        return jsonify({"error":"Provide 'features' (single) or 'instances' (batch)"}), 400

    results = []
    for s in samples:
        try:
            x = preprocess_sample(s)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        pred = int(model.predict(x)[0])
        prob = float(model.predict_proba(x)[0][1])
        results.append({
            "prediction": pred,
            "label": "Fraud" if pred == 1 else "Not Fraud",
            "fraud_probability": prob
        })

    return jsonify(results[0] if len(results) == 1 else {"results": results})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000,debug=True)
