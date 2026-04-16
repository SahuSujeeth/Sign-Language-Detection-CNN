import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from collections import deque
import pyttsx3

# Load model
model = load_model("model.h5")

with open("labels.pkl", "rb") as f:
    le = pickle.load(f)

# 🔊 Initialize voice engine
engine = pyttsx3.init()
last_spoken = ""

cap = cv2.VideoCapture(0)

# 🔥 Stability buffer
buffer = deque(maxlen=7)

mode = "alphabet"

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ROI box
    cv2.rectangle(frame, (100, 100), (350, 350), (0, 255, 0), 2)

    roi = frame[100:350, 100:350]

    # Preprocess
    img = cv2.resize(roi, (64,64))
    img = img / 255.0
    img = np.reshape(img, (1,64,64,3))

    prediction = model.predict(img, verbose=0)

    confidence = np.max(prediction)
    label = le.inverse_transform([np.argmax(prediction)])[0]

    # 🔥 Mode filter
    if mode == "alphabet" and not label.isalpha():
        label = "Invalid"
    elif mode == "number" and not label.isdigit():
        label = "Invalid"

    # 🔥 Error handling + stability
    if confidence < 0.75 or label == "Invalid":
        final_label = "No Sign"
    else:
        buffer.append(label)
        final_label = max(set(buffer), key=buffer.count)

    text = f"{final_label} ({confidence*100:.2f}%)"

    # 🔊 Voice output
    if final_label != "No Sign" and final_label != last_spoken:
        engine.say(final_label)
        engine.runAndWait()
        last_spoken = final_label

    # Display mode
    cv2.putText(frame, f"Mode: {mode}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    # Display prediction
    cv2.putText(frame, text, (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Sign Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    # 🔥 Mode switching
    if key == ord('a'):
        mode = "alphabet"
        buffer.clear()

    elif key == ord('n'):
        mode = "number"
        buffer.clear()

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()