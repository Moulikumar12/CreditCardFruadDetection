import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    return df

def basic_eda(df):
    print("First 5 rows:\n", df.head())
    print("Class distribution:\n", df['Class'].value_counts())

def scale_time_amount(df, scaler_path=None):
    scaler = StandardScaler()
    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
    if scaler_path:
        joblib.dump(scaler, scaler_path)
    return df, scaler

def split_xy(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y
