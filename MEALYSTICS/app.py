from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np

app = FastAPI()

# Model untuk input data
class ColumnInput(BaseModel):
    Total: float
    PaymentMethod: float
    Hour: float
    DayOfWeek: float
    ClusterKMeans: int = None  # Ini bisa diabaikan saat prediksi
    ClusterHC: int = None  # Ini bisa diabaikan saat prediksi
    PCA1: float = None  # Ini bisa diabaikan saat prediksi
    PCA2: float = None  # Ini bisa diabaikan saat prediksi


# Load model KMeans dan scaler yang sudah dilatih dengan 4 fitur
with open("kmeans_model.pkl", "rb") as file:
    kmeans_model = pickle.load(file)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Fitur yang dipakai saat training (hanya 4 fitur)
feature_columns = ["Total", "PaymentMethod", "Hour", "DayOfWeek", "ClusterKMeans", "ClusterHC", "PCA1", "PCA2"]

@app.post("/predict")
def predict(input_data: ColumnInput):
    try:
        # Konversi input ke DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Pilih hanya 4 fitur yang digunakan saat pelatihan
        features = input_df[feature_columns]
        features = features[["Total", "PaymentMethod", "Hour", "DayOfWeek", "ClusterKMeans", "ClusterHC", "PCA1", "PCA2"]]

        # Pastikan format input untuk prediksi sesuai (2D array)
        features_array = features.values  # Ini mengubah DataFrame ke numpy array 2D

        # Lakukan prediksi dengan model KMeans
        cluster = kmeans_model.predict(features_array)

        # Return response yang jelas dan format JSON
        return {
            "cluster": int(cluster[0]),
            "message": f"Data termasuk ke dalam Cluster {cluster[0]}"
        }

    except Exception as e:
        # Tangani error dan kembalikan pesan error dalam format JSON
        return {"error": str(e)}

