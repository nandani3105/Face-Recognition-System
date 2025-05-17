import tkinter as tk
from tkinter import messagebox
import os
import cv2
import shutil
import subprocess
from datetime import datetime

DATASET_DIR = 'dataset'

# Create dataset folder if it doesn't exist
os.makedirs(DATASET_DIR, exist_ok=True)

def capture_face():
    name = name_entry.get()
    if not name:
        messagebox.showwarning("Input Error", "Enter a name before capturing.")
        return

    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    while count < 5:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Capturing - Press 'q' to stop", frame)
        img_path = os.path.join(person_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Success", f"Captured {count} images for {name}.")

def train_model():
    subprocess.run(["python", "train_embeddings.py"])
    messagebox.showinfo("Training", "Embeddings trained successfully!")

def recognize_faces():
    subprocess.run(["python", "recognize.py"])

def exit_app():
    root.destroy()

# GUI Layout
root = tk.Tk()
root.title("Face Recognition System")
root.geometry("400x300")

tk.Label(root, text="Enter Name:", font=('Arial', 12)).pack(pady=10)
name_entry = tk.Entry(root, font=('Arial', 12))
name_entry.pack()

tk.Button(root, text="Capture Face", command=capture_face, font=('Arial', 12)).pack(pady=10)
tk.Button(root, text="Train Embeddings", command=train_model, font=('Arial', 12)).pack(pady=10)
tk.Button(root, text="Start Recognition", command=recognize_faces, font=('Arial', 12)).pack(pady=10)
tk.Button(root, text="Exit", command=exit_app, font=('Arial', 12)).pack(pady=10)

root.mainloop()
