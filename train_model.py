import cv2
import numpy as np
import os

# Path to your dataset
dataset_dir = r"C:\PS 2\final final codes\Face Dataset\Faces"

recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8, threshold=80)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

current_id = 0
label_ids = {}
training_faces = []
labels = []

print("🧠 Starting training...")

# Load dataset and map labels
for category in os.listdir(dataset_dir):
    category_path = os.path.join(dataset_dir, category)
    if not os.path.isdir(category_path):
        continue

    for person in os.listdir(category_path):
        person_path = os.path.join(category_path, person)
        if not os.path.isdir(person_path):
            continue

        label = f"{person}|{category}"
        if label not in label_ids:
            label_ids[label] = current_id
            current_id += 1

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                roi = image[y:y+h, x:x+w]
                training_faces.append(roi)
                labels.append(label_ids[label])

print(f"🖼️ Total faces: {len(training_faces)}")
print(f"👤 Unique labels: {len(label_ids)}")

# Train recognizer with the faces
if training_faces:
    recognizer.train(training_faces, np.array(labels))
    recognizer.save("trained_face_model.yml")
    np.save("labels.npy", {v: k.split('|') for k, v in label_ids.items()})
    print("✅ Training completed. Model and labels saved.")
else:
    print("❌ No faces found. Please check your dataset.")
