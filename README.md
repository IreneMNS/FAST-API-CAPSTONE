

# 📊Analisis Segmentasi Pelanggan Menggunakan Metode Unsupervised Learning Meningkatkan Engagements Waroenk Kangmas.



---

## 📁 Struktur File
```
├── app.py                  # Source code utama API menggunakan FastAPI
├── kmeans_model.pkl        # Model Kmeans (Algoritma terbaik)
├── scaler.pkl               # Scaler untuk preprocessing data (pickle file)
├── requirements.txt         # Daftar dependencies untuk environment
```

---

## 💻 Teknologi yang Digunakan
- Google Colab
- FastAPI
- Scikit-learn
- Kmeans
- Pandas
- Numpy
- Pickle

---

## 🚀 Cara Menjalankan API

1. **Clone repository ini:**
   ```bash
   git clone https://github.com/IreneMNS/FAST-API-CAPSTONE-.git
   cd FAST-API-CAPSTONE
   ```

2. **Buat dan aktifkan virtual environment (opsional tapi direkomendasikan):**
   ```bash
   python -m venv env
   source env/bin/activate  
   env\Scripts\activate     
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi FastAPI:**
   ```bash
   uvicorn app:app --reload
   ```

5. **Akses API di browser:**
   - [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📚 Source Code (app.py) 
```python
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
