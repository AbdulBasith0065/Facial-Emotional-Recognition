import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import defaultdict


@st.cache_resource
def get_model():
    return load_model("emotion_class.h5", compile=True)

model = get_model()
# Load the emotion detection model
# model = load_model("emotion_class.h5",compile=True)

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
emotions = ['happy', 'sad', 'neutral']
img_size = 48

# Streamlit UI
st.title("😊 Facial Emotion Detection")

# Session state initialization
if "run" not in st.session_state:
    st.session_state.run = False
if "emotion_counter" not in st.session_state:
    st.session_state.emotion_counter = defaultdict(int)
if "total_frames" not in st.session_state:
    st.session_state.total_frames = 0

# Control buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Webcam", key="start_button"):
        st.session_state.run = True
        st.session_state.emotion_counter = defaultdict(int)
        st.session_state.total_frames = 0
with col2:
    if st.button("⏹ Stop", key="stop_button"):
        st.session_state.run = False

# Webcam and emotion detection loop
frame_placeholder = st.empty()
if st.session_state.run:
    cap = cv2.VideoCapture(0)

    while cap.isOpened() and st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (img_size, img_size))
            roi = roi.reshape(1, img_size, img_size, 1) / 255.0

            prediction = model.predict(roi, verbose=0)
            emotion = emotions[np.argmax(prediction)]
            st.session_state.emotion_counter[emotion] += 1
            st.session_state.total_frames += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    cv2.destroyAllWindows()
    st.session_state.run = False

# Display results after stopping
if not st.session_state.run and st.session_state.total_frames > 0:
    st.subheader("📊 Emotion Analysis Summary")
    for emotion in emotions:
        count = st.session_state.emotion_counter[emotion]
        try:
            percent = (count / st.session_state.total_frames) * 100
        except ZeroDivisionError:
            percent = 0
        st.write(f"**{emotion.capitalize()}**: {percent:.2f}%")

    # Optional chart
    st.bar_chart({e: st.session_state.emotion_counter[e] for e in emotions})
