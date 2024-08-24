# import time
# import cv2
# import numpy as np
# from paddleocr import PaddleOCR, draw_ocr
# import matplotlib.pyplot as plt

# # Initialize the PaddleOCR model
# ocr = PaddleOCR(use_angle_cls=True, lang="en")

# # Load the video
# video_path = "/home/shusrith/vid.mp4"
# cap = cv2.VideoCapture(video_path)


# # Function to extract and read each section from a frame
# def read_section(frame, contour):
#     x, y, w, h = cv2.boundingRect(contour)
#     section_image = frame[y : y + h, x : x + w]

#     # Use PaddleOCR to read the section
#     result = ocr.ocr(section_image, cls=True)
#     detected_text = ""
#     if result and isinstance(result, list):
#         for idx in range(len(result)):
#             res = result[idx]
#             if res:
#                 for line in res:
#                     detected_text = " " + line[1][0]
#                     bbox = line[0]
#                     print(detected_text)
#                 cv2.polylines(
#                     section_image,
#                     [np.array(bbox).astype(np.int32)],
#                     True,
#                     (0, 255, 0),
#                     2,
#                 )

#         return section_image, result
#     else:
#         return section_image, []


# cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
# frame_count = 0
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     if frame_count % 15 != 0:
#         frame_count += 1
#         continue

#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Apply thresholding to get a binary image
#     _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

#     # Find contours
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Sort contours top-to-bottom and left-to-right
#     contours = sorted(
#         contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0])
#     )

#     # Read each section
#     for contour in contours:
#         # Filter out small contours
#         if cv2.contourArea(contour) < 10000:
#             continue

#         section_image, result = read_section(frame, contour)

#         cv2.resizeWindow("frame", 800, 600)
#         cv2.waitKey(1)
#         cv2.imshow("frame", section_image)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#         time.sleep(1)
#     time.sleep(1)
# # Release the video capture and close windows
# cap.release()
# cv2.destroyAllWindows()


# from paddleocr import PaddleOCR, draw_ocr
# import cv2
# import logging
# import numpy as np

# logging.getLogger("ppocr").setLevel(logging.ERROR)


# ocr = PaddleOCR(use_angle_cls=True, lang="en")

# img_path = "/home/shusrith/projects/blind-eyes/blind-eye-dealers/20240722_181421.jpg"

# img = cv2.imread(img_path)
# result = ocr.ocr(img, cls=True)

# detected_text = ""
# for idx in range(len(result)):
#     res = result[idx]
#     if res:
#         for line in res:
#             detected_text = " " + line[1][0]
#             bbox = line[0]
#             points = np.array(bbox).astype(np.int32).reshape((-1, 1, 2))
#             cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
#             print(bbox)
#             print(detected_text)


# cv2.imshow("Detected Text", img)
# if cv2.waitKey(0) & 0xFF == ord("q"):  # Wait until 'q' is pressed
#     cv2.destroyAllWindows()


import cv2
import numpy as np


image_path = (
    "image.jpg"
)
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

thresh = cv2.bitwise_not(thresh)

vertical_projection = np.sum(thresh, axis=0)

middle_x = np.argmax(vertical_projection)

confidence_threshold = 0.95 * np.max(vertical_projection)
if vertical_projection[middle_x] < confidence_threshold:
    middle_x = -1  #
if middle_x != -1:
    left_page = image[:, :middle_x]
    right_page = image[:, middle_x:]

    cv2.imwrite("left_page.jpg", left_page)
    cv2.imwrite("right_page.jpg", right_page)

cv2.imshow("thresh", left_page)
cv2.waitKey(0)
