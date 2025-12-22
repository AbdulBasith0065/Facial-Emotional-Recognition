import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained CNN model
model = load_model('emotion_class.h5',compile=True)

# Define emotion labels (must match training order)
emotions = ['happy', 'sad', 'neutral']

# Image size used during training
img_size = 48

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Crop and preprocess the face
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (img_size, img_size))
        face_roi = face_roi.reshape(1, img_size, img_size, 1) / 255.0
        
        # Predict emotion
        prediction = model.predict(face_roi, verbose=0)
        emotion_label = emotions[np.argmax(prediction)]
        
        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)
    
    cv2.imshow('Emotion Detection (Happy/Sad/Neutral)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(tf.__version__)