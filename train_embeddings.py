import os
import cv2
import pickle
from utils import get_face_embedding

dataset_path = 'dataset'
embeddings = []
names = []

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        image = cv2.imread(image_path)
        embedding = get_face_embedding(image)
        if embedding is not None:
            embeddings.append(embedding)
            names.append(person_name)

# Save embeddings
with open('dataset/data.pkl', 'wb') as f:
    pickle.dump((embeddings, names), f)

print("Embeddings trained and saved.")
