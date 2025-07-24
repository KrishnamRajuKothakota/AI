import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_and_preprocess_data(file_path, target_column, seq_length=7):
    df = pd.read_csv(file_path)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Encode categorical values
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    feature_columns = [col for col in df.columns if col != target_column]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_columns + [target_column]])

    X, y = [], []
    for i in range(len(scaled) - seq_length):
        X.append(scaled[i:i + seq_length, :-1])
        y.append(scaled[i + seq_length, -1])

    return np.array(X), np.array(y), scaler, df.columns.tolist(), label_encoders
