import cv2
import numpy as np
import math
from ultralytics import YOLO

# Load YOLO model
model = YOLO("/home/shusrith/projects/blind-eye-dealers/yolov8n.pt")

# Known parameters
KNOWN_HEIGHT = 1.0  # Example: height of a known object in meters
FOCAL_LENGTH = (
    949  # Example: focal length in pixels (should be calibrated for your camera)
)


def measure_distance(h):
    distance = (KNOWN_HEIGHT * FOCAL_LENGTH) / h
    return distance / 3


cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 10 != 0:
        continue

    # Get the results from the YOLO model
    results = model(frame)

    # Loop through the detected objects
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            h = y2 - y1
            distance = measure_distance(h)
            distance_str = f"{distance:.2f} meters"

            # Define the bottom left starting point for the text (x, y)
            text_start = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)

            # Draw the bounding box and put the text on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                distance_str,
                text_start,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

    # Display the frame
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
