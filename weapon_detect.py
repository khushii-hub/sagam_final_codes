import cv2
import numpy as np
import os

prototxt = "MobileNetSSD_deploy.prototxt"
model = "MobileNetSSD_deploy.caffemodel"

if not os.path.exists(prototxt) or not os.path.exists(model):
    print("Error: Model files not found")
    exit()

net = cv2.dnn.readNetFromCaffe(prototxt, model)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "knife", "gun"]  # Simulated extra classes

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access camera")
    exit()

print("[INFO] Weapon detection started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if idx >= len(CLASSES):
                continue
            label = CLASSES[idx]
            if label in ["knife", "gun", "person"]:  # Simulating weapon class
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Weapon Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
