# src/explain.py
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

def explain_model(model_path, sample_csv):
    model = joblib.load(model_path)
    X = pd.read_csv(sample_csv)
    # load smaller sample to visualize
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("artifacts/shap_summary.png")
    plt.close()
    # local force plot for first instance (matplotlib)
    shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:], matplotlib=True)
    plt.savefig("artifacts/shap_force_0.png")
    plt.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python explain.py <model_path> <sample_csv>")
    else:
        explain_model(sys.argv[1], sys.argv[2])
