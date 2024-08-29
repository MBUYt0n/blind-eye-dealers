import cv2
import numpy as np
from paddleocr import PaddleOCR

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


# Function to separate columns within each section
def separate_columns(section):
    gray = cv2.cvtColor(section, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = np.ones((50, 1), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    columns = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50:  # Filter out small and non-vertical contours
            column = section[y : y + h, x : x + w]
            columns.append(
                (x, y, w, h)
            )  # Store the bounding box instead of the cropped column
    return columns


# Open video capture
vid = cv2.VideoCapture(r"C:\Users\swath\Downloads\blind-eyes\vids\20240824_143156.mp4")
frame_count = 0

while True:
    ret, image = vid.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 6000 != 0:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 3: Apply binary threshold to highlight text
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Step 4: Apply morphological operations to detect large blank spaces
        kernel = np.ones((4, 4), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=3)

        # Step 5: Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Step 6: Initialize an array to store each section
        sections = []

        # Step 7: Loop through the contours to crop and store each section
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            section = image[y : y + h, x : x + w]  # Crop the section
            sections.append(
                (x, y, w, h)
            )  # Store the bounding box instead of the cropped section

        # Step 8: Separate columns within each section and draw bounding boxes
        for idx, (sx, sy, sw, sh) in enumerate(sections):
            section = image[sy : sy + sh, sx : sx + sw]
            columns = separate_columns(section)
            for col_idx, (cx, cy, cw, ch) in enumerate(columns):
                try:
                    col = section[cy : cy + ch, cx : cx + cw]
                    res = ocr.ocr(col, cls=True)
                    text = " ".join([i[1][0] for i in res[0]])
                    print(f"Section {idx+1} - Column {col_idx+1}: {text}")
                    # Draw bounding box around the column
                    cv2.rectangle(
                        image,
                        (sx + cx, sy + cy),
                        (sx + cx + cw, sy + cy + ch),
                        (0, 255, 0),
                        2,
                    )
                except:
                    continue

        # Display the overall frame with bounding boxes
        cv2.imshow("Frame with Columns", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

vid.release()
cv2.destroyAllWindows()
