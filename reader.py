import time
import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import matplotlib.pyplot as plt

# Initialize the PaddleOCR model
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Load the video
video_path = "/home/shusrith/vid.mp4"
cap = cv2.VideoCapture(video_path)


# Function to extract and read each section from a frame
def read_section(frame, contour):
    x, y, w, h = cv2.boundingRect(contour)
    section_image = frame[y : y + h, x : x + w]

    # Use PaddleOCR to read the section
    result = ocr.ocr(section_image, cls=True)
    detected_text = ""
    if result and isinstance(result, list):
        for idx in range(len(result)):
            res = result[idx]
            if res:
                for line in res:
                    detected_text = " " + line[1][0]
                    bbox = line[0]
                    print(detected_text)
                cv2.polylines(
                    section_image,
                    [np.array(bbox).astype(np.int32)],
                    True,
                    (0, 255, 0),
                    2,
                )

        return section_image, result
    else:
        return section_image, []


cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 15 != 0:
        frame_count += 1
        continue
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get a binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours top-to-bottom and left-to-right
    contours = sorted(
        contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0])
    )

    # Read each section
    for contour in contours:
        # Filter out small contours
        if cv2.contourArea(contour) < 10000:
            continue

        section_image, result = read_section(frame, contour)

        cv2.resizeWindow("frame", 800, 600)
        cv2.waitKey(1)
        cv2.imshow("frame", section_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        time.sleep(1)
    time.sleep(1)
# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
