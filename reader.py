import cv2
from paddleocr import PaddleOCR
import numpy as np

ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


def cosine(a, b):
    max_len = max(len(a), len(b))
    a = np.pad(a, (0, max_len - len(a)), mode="constant")
    b = np.pad(b, (0, max_len - len(b)), mode="constant")
    cosVal = (np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cosVal


vid = cv2.VideoCapture(r"C:\Users\swath\Downloads\blind-eyes\vids\book.mp4")
frame_count = 0
prev = []

prevCos = 0
currCos = 0
while True:
    ret, frame = vid.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 1000 != 0:
        curr = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = ocr.ocr(gray, cls=True)
        text = ""
        if result[0] is not None:
            for i in result[0]:
                _, t = i
                curr.append(sum([ord(i) for i in t[0]]))
                text += " " + t[0]
            curr = np.array(curr)
            currCos = cosine(prev, curr)
            prevCos = currCos
            if prevCos > 0.85 and currCos > 0.85:
                print(text, currCos)
            prev = [i for i in curr]
            prev = np.array(prev)

        cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break