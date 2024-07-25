import cv2


def capture_stereo_images():
    left_camera = cv2.VideoCapture(0)
    right_camera = cv2.VideoCapture(1)

    ret_left, frame_left = left_camera.read()
    ret_right, frame_right = right_camera.read()

    left_camera.release()
    right_camera.release()

    if ret_left and ret_right:
        cv2.imwrite("left_image.jpg", frame_left)
        cv2.imwrite("right_image.jpg", frame_right)
    else:
        print("Failed to capture images")


capture_stereo_images()


def compute_disparity(left_img, right_img):
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoBM_create(numDisparities=16 * 10, blockSize=15)
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    return disparity


left_img = cv2.imread("left_image.jpg")
right_img = cv2.imread("right_image.jpg")

disparity_map = compute_disparity(left_img, right_img)
cv2.imshow("Disparity Map", disparity_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
import numpy as np


def calculate_depth(disparity_map, B=0.06, focal_length=700):
    # B is the baseline distance between the cameras (in meters)
    # focal_length is in pixels (approximated, can be estimated from camera specifications)

    # Avoid division by zero by setting disparity values of zero to a very small number
    disparity_map[disparity_map == 0] = 0.1

    # Calculate depth
    depth_map = (B * focal_length) / disparity_map

    return depth_map


depth_map = calculate_depth(disparity_map)
cv2.imshow("Depth Map", depth_map / np.max(depth_map))  # Normalize for visualization
cv2.waitKey(0)
cv2.destroyAllWindows()


def identify_roi(depth_map, threshold=0.2):
    mean_depth = np.mean(depth_map)
    mask = np.abs(depth_map - mean_depth) < threshold * mean_depth
    mask = mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        roi = (x, y, w, h)
    else:
        roi = (0, 0, depth_map.shape[1], depth_map.shape[0])

    return roi


roi = identify_roi(depth_map)
x, y, w, h = roi
cropped_image = left_img[y : y + h, x : x + w]
cv2.imshow("Cropped Image", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
from paddleocr import PaddleOCR


def detect_text(image_path):
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    result = ocr.ocr(image_path, cls=True)
    return result


cv2.imwrite("cropped_image.jpg", cropped_image)
text_result = detect_text("cropped_image.jpg")


def extract_headlines(text_result):
    headlines = []
    for line in text_result:
        bbox = line[0]
        text = line[1][0]
        if bbox[1][1] - bbox[0][1] > 50 and bbox[0][1] < 500:
            headlines.append(text)
    return headlines


headlines = extract_headlines(text_result)
for i, headline in enumerate(headlines):
    print(f"Headline {i+1}: {headline}")


def on_button_press():
    selected_headline = simpledialog.askinteger(
        "Input", "Enter the headline number to read the full article"
    )
    if selected_headline is not None and 1 <= selected_headline <= len(headlines):
        read_full_article(headlines[selected_headline - 1])


def extract_article(text_result, headline):
    article = ""
    for line in text_result:
        text = line[1][0]
        if text.startswith(headline):
            article += text + " "
    return article


def read_full_article(headline):
    article = extract_article(text_result, headline)
    read_text(article)


def read_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


if button_pressed:
    on_button_press()
