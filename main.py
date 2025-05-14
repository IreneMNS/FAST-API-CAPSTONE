from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np
# FastAPI untuk membuat API 


app = FastAPI()

# Model untuk input data
class ColumnInput(BaseModel):
    Total: float
    PaymentMethod: float
    Hour: float
    DayOfWeek: float


# Load model KMeans dan scaler yang sudah dilatih dengan 4 fitur
with open("kmeans_model.pkl", "rb") as file:
    kmeans_model = pickle.load(file)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Fitur yang dipakai saat training (hanya 4 fitur)
feature_columns = ["Total", "PaymentMethod", "Hour", "DayOfWeek"]

@app.post("/predict")
def predict(input_data: ColumnInput):
    try:
        # Konversi input ke DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Pilih hanya 4 fitur yang digunakan saat pelatihan
        features = input_df[feature_columns]
        features = features[["Total", "PaymentMethod", "Hour", "DayOfWeek"]]
        print(features)

        # Pastikan format input untuk prediksi sesuai (2D array)
        features_array = features.values  # Ini mengubah DataFrame ke numpy array 2D

        # Lakukan prediksi dengan model KMeans
        cluster = kmeans_model.predict(features_array)
        print(cluster)

        # Return response yang jelas dan format JSON
        return {
            "cluster": int(cluster[0]),
            "message": f"Data termasuk ke dalam Cluster {cluster[0]}"
        }

    except Exception as e:
        # Tangani error dan kembalikan pesan error dalam format JSON
        return {"error": str(e)}

@app.get("/")
def home_root():
    return {"message": "Success"}
# Menjalankan server FastAPI
# Gunakan perintah berikut di terminal untuk menjalankan server:
# uvicorn main:app --reload
# Pastikan untuk menginstall FastAPI dan Uvicorn jika belum terinstall
# pip install fastapi uvicorn
# Untuk menguji API, Anda bisa menggunakan Postman atau curl
# Untuk menguji API, Anda bisa menggunakan Postman atau curl
# Contoh curl untuk menguji API
# curl -X POST "http://
# localhost:8000/predict" -H "Content-Type: application/json" -d '{"Total": 100, "PaymentMethod": 1, "Hour": 12, "DayOfWeek": 3}'


@app.get("/deploy")
def home_root():
    return {"message": "Vercel Deployment Success"}
# Menjalankan server FastAPI
# Gunakan perintah berikut di terminal untuk menjalankan server:
# uvicorn main:app --reload
# Pastikan untuk menginstall FastAPI dan Uvicorn jika belum terinstall