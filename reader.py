import cv2
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

image_path = "image.jpg"
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

result = ocr.ocr(gray, cls=True)


def is_headline(bbox):
    y1, y2, y3, y4 = bbox
    y_min = min(y1[1], y2[1], y3[1], y4[1])
    y_max = max(y1[1], y2[1], y3[1], y4[1])
    height = y_max - y_min
    return height > 50


headlines = []
other_texts = []

for line in result[0]:
    bbox, text = line
    if is_headline(bbox):
        headlines.append(text)
    else:
        other_texts.append(text)

print("Headlines:")
for headline in headlines:
    print(headline)

print("\nOther Texts:")
for text in other_texts:
    print(text)
