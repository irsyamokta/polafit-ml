import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATASET_PATH = 'dataset/dataset_olahraga.csv'
MODEL_PATH = 'models/recomendation_sportTf_model.h5'

df = pd.read_csv(DATASET_PATH)

numeric_features = ['Tinggi (cm)', 'Berat (kg)', 'Durasi (menit)', 'Kalori Terbakar', 'Umur']
categorical_features = ['Tingkat Aktivitas', 'Tujuan', 'Kategori', 'Jenis Kelamin']
target_column = 'Rekomendasi Olahraga'

X = df.drop(target_column, axis=1)
y = df[target_column]

X_encoded = pd.get_dummies(X[categorical_features])

scaler = StandardScaler()
X_combined = pd.concat([X[numeric_features], X_encoded], axis=1)
X_combined[numeric_features] = scaler.fit_transform(X_combined[numeric_features])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

def load_exercise_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def predict_exercise(input_data):
    input_encoded = pd.get_dummies(input_data[categorical_features])

    missing_cols = set(X_encoded.columns) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0
    input_encoded = input_encoded[X_encoded.columns]

    input_combined = input_data[numeric_features].copy()
    input_combined[numeric_features] = scaler.transform(input_combined[numeric_features])
    input_combined = pd.concat([input_combined, input_encoded], axis=1)

    # Load model and make predictions
    model = load_exercise_model()
    prediction = model.predict(input_combined)
    predicted_class = np.argmax(prediction, axis=1)

    recommended_exercise = label_encoder.inverse_transform(predicted_class)
    return recommended_exercise[0]
