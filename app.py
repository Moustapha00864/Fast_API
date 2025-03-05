from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from keras.preprocessing import image as i1
from keras import models
import os
import cv2
import numpy as np

app = Flask(__name__)

# Configuration du dossier d'upload
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key='secret1234'


# Charger le modèle
model = models.load_model('static\models\model.h5')

# Vérifier si le fichier est autorisé
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fonction de prédiction
def predict_label(img_path):
    i = cv2.imread(img_path)
    resized = cv2.resize(i, (50, 50))
    i = i1.img_to_array(resized) / 255.0
    i = i.reshape(1, 50, 50, 3)
    result = model.predict(i)
    a, b = round(result[0, 0], 2) * 100, round(result[0, 1], 2) * 100
    threshold = 10

    if a > threshold or b > threshold:
        ind = np.argmax(result)
        classes = ['Cellule Normal: Pas de Paludisme', 'Cellule Infecté :Présence du Paludisme']
        return classes[ind], f"{max(a, b):.2f}%"
    else:
        return 'Invalid Image', "0%"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Aucun fichier sélectionné')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('Aucun fichier sélectionné')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            label, probability = predict_label(filepath)

            return render_template('upload.html', label=label, probability=probability, filename=filename)

    return render_template('upload.html', label=None)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
