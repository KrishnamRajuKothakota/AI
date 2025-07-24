import streamlit as st
import pandas as pd
from model import build_cnn_model
from preprocess import load_and_preprocess_data
from sklearn.metrics import mean_squared_error
import numpy as np
import os

st.title("🌦 Weather Forecast CNN App")

csv_file = "weather.csv"
if not os.path.exists(csv_file):
    st.error(f"CSV file not found: {csv_file}")
    st.stop()

# Load the CSV once to get column options
df_sample = pd.read_csv(csv_file)
target_options = df_sample.columns.tolist()

# User selects the target
target_column = st.selectbox("Select the value to predict", target_options)

if st.button("Train and Predict"):
    with st.spinner("Processing..."):
        # Prepare data
        X, y, scaler, all_columns, encoders = load_and_preprocess_data(csv_file, target_column)
        X_train, X_test = X[:-100], X[-100:]
        y_train, y_test = y[:-100], y[-100:]

        model = build_cnn_model(X_train.shape[1:])
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

        # Predict and show results
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        st.success(f"Model trained. MSE on last 100 samples: {mse:.4f}")

        st.line_chart({"Actual": y_test, "Predicted": predictions.flatten()})
