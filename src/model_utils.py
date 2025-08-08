import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def resample_smote(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

def train_rf(X, y, n_estimators=200):
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf

def train_xgb(X, y, n_estimators=100, max_depth=5):
    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb.fit(X, y)
    return xgb

def save_model(model, path):
    joblib.dump(model, path)
    print(f"Model saved to {path}")
