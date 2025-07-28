import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
import joblib

# Load dataset
data = pd.read_csv('../dataset/nsl_kdd_sample.csv')

# Preprocess
categorical_columns = ['protocol_type', 'service', 'flag']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

X = data.drop('label', axis=1)
y = LabelEncoder().fit_transform(data['label'])
y = to_categorical(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Reshape for CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# CNN model
model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model and scaler
model.save('../streamlit_app/model_cnn.h5')
joblib.dump(scaler, '../streamlit_app/scaler.pkl')
