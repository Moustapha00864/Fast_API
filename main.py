from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
import os
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
def predict_label(img_path):
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail=f"File not found: {img_path}")

    i = cv2.imread(img_path)
    if i is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    resized = cv2.resize(i, (50, 50))
    i = i1.img_to_array(resized) / 255.0
    i = i.reshape(1, 50, 50, 3)
    result = model.predict(i)
    a = round(result[0, 0], 2) * 100
    b = round(result[0, 1], 2) * 100
    probability = [a, b]
    threshold = 10

    if a > threshold or b > threshold:
        ind = np.argmax(result)
        classes = ["Cellule Normal: Pas de Paludisme", "Cellule Infecté :Présence du Paludisme"]
        return classes[ind], probability[ind]
    else:
        return "Invalid Image", 0

# Endpoint pour l'upload et la classification
@app.post("/predict/")
async def classify_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    try:
        # Sauvegarde du fichier
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Vérification si le fichier existe bien après l'enregistrement
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="File upload failed")

        # Classification
        label, probability = predict_label(file_path)
        return {"filename": file.filename, "label": label, "probability": probability}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route d'accueil
@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de classification du paludisme"}
