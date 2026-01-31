from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import tensorflow as tf
import os
from datetime import datetime

# ---------------- APP ----------------
app = FastAPI(title="Smart Home Ear Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- CONSTANTS ----------------
UPLOAD_DIR = "uploads"
MODEL_PATH = "smartear_epoch15_model.h5"

SR = 16000
N_MELS = 128
MAX_LEN = 128

CLASSES = ["danger", "alert", "safe"]


os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ CNN Model Loaded Successfully")

# ---------------- AUDIO → MEL (SAME AS COLAB) ----------------
def wav_to_mel(file_path):
    y, sr = librosa.load(file_path, sr=SR)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=512,
        n_mels=N_MELS
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    if mel.shape[1] < MAX_LEN:
        mel = np.pad(mel, ((0, 0), (0, MAX_LEN - mel.shape[1])))
    else:
        mel = mel[:, :MAX_LEN]

    mel = mel[..., np.newaxis]  # (128,128,1)
    return mel

# ---------------- PREDICTION LOGIC ----------------
def predict_audio(file_path):
    mel = wav_to_mel(file_path)
    mel = np.expand_dims(mel, axis=0)

    preds = model.predict(mel, verbose=0)[0]
    class_id = int(np.argmax(preds))
    confidence = float(preds[class_id]) * 100

    return {
        "time": str(datetime.now()),
        "risk": CLASSES[class_id],
        "sound": CLASSES[class_id],
        "confidence": round(confidence, 2),
        "danger": CLASSES[class_id] == "danger"
    }

# ---------------- API ROUTES ----------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = predict_audio(file_path)
    return result


@app.get("/live-check")
def live_check():
    """
    Demo endpoint for continuous monitoring
    (Frontend polling ke liye)
    """
    # No mic capture on server → frontend mic used
    return {
        "time": str(datetime.now()),
        "risk": "safe",
        "sound": "Background",
        "confidence": 0.0,
        "danger": False
    }


@app.post("/sos")
def sos(data: dict):
    # Hackathon demo logic
    print("🚨 SOS ALERT SENT")
    print(data)

    return {
        "status": "SOS sent",
        "police": data.get("police"),
        "guardian": data.get("guardian")
    }