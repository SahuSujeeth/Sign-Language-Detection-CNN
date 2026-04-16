import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import time

# -----------------------------
# LOAD MODEL
# -----------------------------
model = load_model("model.h5")

with open("labels.pkl", "rb") as f:
    le = pickle.load(f)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Settings")

mode = st.sidebar.radio("Select Mode", ["Alphabet", "Number"])
voice = st.sidebar.checkbox("Enable Voice")

start = st.sidebar.button("▶ Start Camera")
stop = st.sidebar.button("⏹ Stop Camera")

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Model Info")
st.sidebar.success("Model Loaded Successfully")
st.sidebar.write(f"Classes: {len(le.classes_)}")

# -----------------------------
# MAIN TITLE
# -----------------------------
st.title("✋ Sign Language Detection (CNN)")

# Layout
col1, col2 = st.columns([1,2])

# Left side (Prediction info)
with col1:
    st.subheader("🔍 Prediction")
    pred_text = st.empty()
    conf_text = st.empty()
    status_text = st.empty()
    fps_text = st.empty()

# Right side (Camera)
with col2:
    st.subheader("📷 Live Camera Feed")
    frame_window = st.image([])

# -----------------------------
# CAMERA LOOP
# -----------------------------
run = False

if start:
    run = True

if stop:
    run = False

cap = cv2.VideoCapture(0)

prev_time = 0

while run:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ROI box
    cv2.rectangle(frame, (100, 100), (350, 350), (0,255,0), 2)
    roi = frame[100:350, 100:350]

    # Preprocess
    img = cv2.resize(roi, (64,64))
    img = img / 255.0
    img = np.reshape(img, (1,64,64,3))

    # Prediction
    prediction = model.predict(img, verbose=0)
    confidence = np.max(prediction)
    label = le.inverse_transform([np.argmax(prediction)])[0]

    # Mode filtering
    if mode == "Alphabet" and not label.isalpha():
        label = "No Sign"
    elif mode == "Number" and not label.isdigit():
        label = "No Sign"

    # Threshold
    if confidence < 0.7:
        label = "No Sign"

    # Status
    status = "✋ Hand Detected" if label != "No Sign" else "❌ Not Detected"

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Display on frame
    cv2.putText(frame, f"{label} ({confidence*100:.2f}%)",
                (100, 90), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    # Update UI
    pred_text.markdown(f"### {label}")
    conf_text.write(f"Confidence: {confidence:.2f}")
    status_text.write(f"Status: {status}")
    fps_text.write(f"FPS: {fps:.2f}")

    frame_window.image(frame, channels="BGR")

cap.release()