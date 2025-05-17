import cv2
import numpy as np
from keras_facenet import FaceNet

embedder = FaceNet()

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.3, 5)

def get_face_embedding(image):
    faces = embedder.extract(image, threshold=0.95)
    if faces:
        return faces[0]['embedding']
    return None
