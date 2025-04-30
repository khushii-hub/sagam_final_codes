import cv2
import numpy as np
import os

MODEL_PATH = "trained_face_model.yml"
LABELS_PATH = "labels.npy"

def detect_faces(source=0):
    print("🔄 Loading model and labels...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print("❌ Model or labels not found. Train first using train_model.py.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    label_map = np.load(LABELS_PATH, allow_pickle=True).item()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Unable to open video source.")
        return

    print("✅ Detection started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame. Exiting.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        print(f"🧠 Detected {len(faces)} face(s)")

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            label_id, confidence = recognizer.predict(roi_gray)
            name, category = label_map.get(label_id, ("Unknown", "unknown"))

            color = (0, 0, 255) if category == 'criminals' else (0, 255, 0)
            text = f"{name} ({category})"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display the frame
        cv2.imshow("Face Detection", frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Exiting detection.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change to video file path if needed
    detect_faces(0)  # Webcam
