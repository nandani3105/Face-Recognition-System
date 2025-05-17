import cv2
import pickle
import numpy as np
from utils import get_face_embedding
from sklearn.metrics.pairwise import cosine_similarity

with open('dataset/data.pkl', 'rb') as f:
    stored_embeddings, names = pickle.load(f)

cap = cv2.VideoCapture(0)

def recognize_face(frame):
    embedding = get_face_embedding(frame)
    if embedding is None:
        return "No Face"
    
    stored_embeddings,
    if not stored_embeddings:
        return "No data available"

    sims = cosine_similarity([embedding], stored_embeddings)
    # max_index = np.argmax(sims)
    # if sims[0][max_index] > 0.6:
    #     return names[max_index]
    # else:
    best_match_index = np.argmax(sims[0])
    best_score = sims[0][best_match_index]

    if best_score > 0.6:
        return names[best_match_index]
    else:
        return "Unknown"
        

while True:
    ret, frame = cap.read()
    if not ret:
        break

    name = recognize_face(frame)
    cv2.putText(frame, name, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
