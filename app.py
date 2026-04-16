from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)
CORS(app)

# Load model
model = load_model("model.h5")

with open("labels.pkl", "rb") as f:
    le = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]

    # Decode image
    img_data = base64.b64decode(data)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Resize same as training
    img = cv2.resize(img, (64,64))
    img = img / 255.0
    img = np.reshape(img, (1,64,64,3))

    prediction = model.predict(img, verbose=0)

    confidence = float(np.max(prediction))
    label = le.inverse_transform([np.argmax(prediction)])[0]

    # 🔥 Confidence filter
    if confidence < 0.75:
        label = "No Sign"

    return jsonify({
        "label": label,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)