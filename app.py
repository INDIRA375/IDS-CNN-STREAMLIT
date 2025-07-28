import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.title("Intrusion Detection System Using CNN")

uploaded_file = st.file_uploader("Upload CSV File", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    model = load_model("model_cnn.h5")
    scaler = joblib.load("scaler.pkl")

    # Encode categorical columns as in training
    categorical_columns = ['protocol_type', 'service', 'flag']
    for col in categorical_columns:
        data[col] = data[col].astype('category').cat.codes

    X = scaler.transform(data)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    predictions = model.predict(X)
    pred_labels = np.argmax(predictions, axis=1)

    st.write("Predicted Labels:")
    st.write(pred_labels)
