from flask import Flask, request, render_template, send_from_directory
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import pickle
import os
import webbrowser
from threading import Timer

# ----------------------
# Load model & encoder
# ----------------------
model_path = os.path.join("..", "models", "skin_disease_cnn.keras")
encoder_path = os.path.join("..", "encoder.pkl")

model = load_model(model_path)
with open(encoder_path, "rb") as f:
    encoder = pickle.load(f)

img_size = 128
UPLOAD_FOLDER = os.path.join("..", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder=UPLOAD_FOLDER)

# ----------------------
# Mapping codes → full names
# ----------------------
disease_names = {
    "nv": "Melanocytic Nevi (Benign Mole)",
    "mel": "Melanoma (Malignant)",
    "bkl": "Benign Keratosis-like Lesion",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratoses",
    "vasc": "Vascular Lesion",
    "df": "Dermatofibroma"
}

malignant_classes = ["mel", "bcc", "akiec"]

# ----------------------
# Prediction function
# ----------------------
def predict_image(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_input = np.expand_dims(img_resized / 255.0, axis=0)
    pred = model.predict(img_input)[0]
    top_indices = pred.argsort()[-3:][::-1]
    results = []
    for i in top_indices:
        code = encoder.classes_[i]
        results.append({
            "name": disease_names[code],
            "prob": float(pred[i]),
            "malignant": code in malignant_classes
        })
    return results

# ----------------------
# Flask routes
# ----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_filename = None
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        image_filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, image_filename)
        file.save(file_path)
        result = predict_image(file_path)
    return render_template("index.html", result=result, image_filename=image_filename)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ----------------------
# Open browser automatically
# ----------------------
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=True)
