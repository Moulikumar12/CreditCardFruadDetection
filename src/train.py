import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import load_data, basic_eda, scale_time_amount, split_xy
from model_utils import resample_smote, train_rf, train_xgb, save_model

OUT_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

def plot_and_save_confusion(cm, title, path):
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    df = load_data("data/creditcard.csv")
    basic_eda(df)

    df, scaler = scale_time_amount(df, scaler_path=f"{OUT_DIR}/scaler.joblib")
    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_res, y_res = resample_smote(X_train, y_train)
    print("After SMOTE:", np.bincount(y_res))

    rf = train_rf(X_res, y_res)
    save_model(rf, f"{OUT_DIR}/rf_model.joblib")

    xgb = train_xgb(X_res, y_res)
    save_model(xgb, f"{OUT_DIR}/xgb_model.joblib")

    for name, model in [("RandomForest", rf), ("XGBoost", xgb)]:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        print("====", name, "====")
        print(classification_report(y_test, y_pred, digits=4))
        print("ROC AUC:", roc_auc_score(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred)
        plot_and_save_confusion(cm, f"{name} Confusion Matrix", f"{OUT_DIR}/{name}_confusion.png")

    precision, recall, _ = precision_recall_curve(y_test, xgb.predict_proba(X_test)[:,1])
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (XGBoost)")
    plt.savefig("artifacts/confusion_matrix_rf.png")
    plt.savefig(f"{OUT_DIR}/xgb_pr_curve.png")
    plt.close()

if __name__ == "__main__":
    main()
