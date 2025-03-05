from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
import os
import io
import cv2
import numpy as np
from keras.preprocessing import image as i1
from keras import models

app = FastAPI()

# Charger le modèle
model = models.load_model("static/models/model.h5")

# Chemin du dossier d'uploads
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Création du dossier s'il n'existe pas

# Fonction de classification
def predict_label(img_bytes):
    # Conversion de l'image en tableau NumPy
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # Redimensionnement et prétraitement
    resized = cv2.resize(img, (50, 50))  
    img_array = i1.img_to_array(resized) / 255.0  # Normalisation
    img_array = img_array.reshape(1, 50, 50, 3)

    # Préparer les tensors pour TensorFlow Lite
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Remplir les entrées du modèle
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))

    # Exécuter l'inférence
    interpreter.invoke()

    # Obtenir les résultats
    result = interpreter.get_tensor(output_details[0]['index'])

    # Convertir les résultats en probabilités (convertir en `float` Python)
    a = float(round(result[0, 0], 2) * 100)
    b = float(round(result[0, 1], 2) * 100)
    probability = [a, b]
    threshold = 10

    if a > threshold or b > threshold:
        ind = int(np.argmax(result))  # Convertir `numpy.int64` en `int`
        classes = ["Cellule Normal: Pas de Paludisme", "Cellule Infectée: Présence du Paludisme"]
        return classes[ind], probability[ind]  # Retourne `float` Python
    else:
        return "Image invalide", 0.0  # `0.0` au lieu de `0`

# Route d'accueil
@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de classification du paludisme"}
