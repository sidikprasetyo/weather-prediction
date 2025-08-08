import firebase_admin
from firebase_admin import credentials, firestore
import joblib
import numpy as np
import schedule
import time
from datetime import datetime

# === 🔐 Inisialisasi Firebase ===
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# === 🧠 Load model dan encoder ===
model = joblib.load("weather_prediction.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# === 📥 Ambil data terbaru dari Firestore ===
def get_latest_sensor_data():
    docs = db.collection("dataHistoryPTLM") \
             .order_by("__name__", direction=firestore.Query.DESCENDING) \
             .limit(1).stream()

    for doc in docs:
        data = doc.to_dict()
        print("📥 Data dari Firestore:", data)

        sensor_input = [
            float(data.get("curah_hujan", 0)),
            float(data.get("kecepatan_angin", 0)),
            float(data.get("kelembaban_udara", 0)),
            float(data.get("radiasi", 0)),
            float(data.get("suhu_udara", 0)),
        ]
        return sensor_input, data

    print("❌ Tidak ada data ditemukan.")
    return None, None

# === 🔮 Prediksi cuaca ===
def predict_weather(sensor_input):
    input_array = np.array([sensor_input])
    prediction = model.predict(input_array)
    return label_encoder.inverse_transform(prediction)[0]

# === 🔁 Jalankan prediksi dan simpan hasilnya ke Firestore ===
def run_prediction_job():
    print(f"🕒 Menjalankan prediksi pada {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    sensor_input, original_data = get_latest_sensor_data()
    if sensor_input:
        prediction = predict_weather(sensor_input)
        print(f"🌤️ Prediksi cuaca: {prediction}")

        # Simpan hasil prediksi ke koleksi "forecasts"
        forecast_data = {
            "timestamp": firestore.SERVER_TIMESTAMP,
            "prediction": prediction,
            "input_data": {
                "curah_hujan": sensor_input[0],
                "kecepatan_angin": sensor_input[1],
                "kelembaban_udara": sensor_input[2],
                "radiasi": sensor_input[3],
                "suhu_udara": sensor_input[4],
            }
        }

        db.collection("forecasts").add(forecast_data)
        print("✅ Hasil prediksi disimpan ke Firestore (koleksi 'forecasts')\n")

# === 🗓️ Jadwalkan setiap 15 menit ===
schedule.every(15).minutes.do(run_prediction_job)

print("📡 Scheduler aktif. Menunggu jadwal prediksi setiap 15 menit...\nTekan Ctrl+C untuk keluar.")

# Loop utama
while True:
    schedule.run_pending()
    time.sleep(1)
