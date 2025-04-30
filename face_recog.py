import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

if not os.path.exists("trained_face_model.yml"):
    print("Error: trained_face_model.yml not found.")
    exit()

recognizer.read("trained_face_model.yml")
labels = {0: "Employee", 1: "Criminal"}  # Adjust based on your data

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access camera")
    exit()

print("[INFO] Face recognition started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(roi)
        name = labels.get(label, "Unknown")
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
