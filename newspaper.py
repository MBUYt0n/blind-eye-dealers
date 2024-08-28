import cv2
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


# Function to separate columns within each section
def separate_columns(section):
    # Use adaptive thresholding instead of a fixed threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Apply morphological operations to detect vertical lines
    kernel = np.ones((50, 1), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=3)

    # Find contours for vertical lines
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    columns = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / float(w)
        if w > 50 and h > 50:  # Filter out small and non-vertical contours
            column = section[y : y + h, x : x + w]  # Crop the column
            columns.append(column)

    return columns


# Step 1: Load the image
image = cv2.imread("image.jpg")

# Step 2: Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply binary threshold to highlight text
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Step 4: Apply morphological operations to detect large blank spaces
kernel = np.ones((4, 4), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=3)

# Step 5: Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 6: Initialize an array to store each section
sections = []

# Step 7: Loop through the contours to crop and store each section
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    section = image[y : y + h, x : x + w]  # Crop the section
    sections.append(section)  # Store the section in the array

# Step 8: Separate columns within each section and display
for idx, sec in enumerate(sections):
    columns = separate_columns(sec)
    for col_idx, col in enumerate(columns):
        try:
            res = ocr.ocr(col, cls=True)
            text = " ".join([i[1][0] for i in res[0]])
            print(f"Section {idx+1} - Column {col_idx+1}: {text}")
        except:
            continue
        cv2.imshow(f"Section {idx+1} - Column {col_idx+1}", col)
        # cv2.waitKey(0)  # Press any key to move to the next column

cv2.destroyAllWindows()
