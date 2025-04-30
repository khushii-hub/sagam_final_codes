import cv2
import numpy as np

def detect_fire():
    """Optimized fire detection with improved color detection, motion detection, and noise reduction."""
    video_capture = cv2.VideoCapture(0)  # Use webcam
    _, prev_frame = video_capture.read()  # Read the first frame for motion detection
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Parameters for temporal consistency
    fire_detected_frames = 0
    required_frames = 5  # Number of consecutive frames to confirm fire

    # Kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)
    
    # Background Subtractor (for improved motion detection)
    background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Refined fire-like colors (red, orange, yellow)
        lower_red = np.array([0, 100, 100])  # Lower bound for red
        upper_red = np.array([10, 255, 255])  # Upper bound for red

        lower_orange = np.array([10, 100, 150])  # Lower bound for orange
        upper_orange = np.array([35, 255, 255])  # Upper bound for orange

        lower_yellow = np.array([20, 150, 150])  # Lower bound for yellow
        upper_yellow = np.array([40, 255, 255])  # Upper bound for yellow

        # Create masks for each fire-like color
        red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)
        orange_mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)
        yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

        # Combine all masks to detect fire-like regions
        fire_mask = cv2.bitwise_or(red_mask, orange_mask)
        fire_mask = cv2.bitwise_or(fire_mask, yellow_mask)

        # Apply morphological operations to reduce noise in the fire mask
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)

        # Motion Detection using Background Subtraction (more accurate than frame differencing)
        fg_mask = background_subtractor.apply(frame)
        
        # Apply morphological operations to clean up the foreground mask
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Combine motion (foreground) and fire masks (only regions with both motion and fire-like colors)
        combined_mask = cv2.bitwise_and(fg_mask, fire_mask)

        # Find contours in the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Check for significant fire regions
        fire_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 2000:  # Adjust area threshold based on your environment
                fire_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw bounding box
                cv2.putText(frame, "Fire Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Temporal consistency check (ensure fire detection persists over multiple frames)
        if fire_detected:
            fire_detected_frames += 1
        else:
            fire_detected_frames = 0

        if fire_detected_frames >= required_frames:
            cv2.putText(frame, "Confirmed Fire", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Update the previous frame for background subtraction
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the frame with detection result
        cv2.imshow("Fire Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_fire()
