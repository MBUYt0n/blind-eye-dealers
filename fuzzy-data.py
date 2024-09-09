import cv2
from ultralytics import YOLO
import logging
from paddleocr import PaddleOCR

logging.getLogger("ultralytics").setLevel(logging.ERROR)


def org(res):
    curr = {}
    for i in res:
        if i[0] not in curr:
            curr[i[0]] = []

        x1, y1, x2, y2 = (
            int(i[1]["x1"]),
            int(i[1]["y1"]),
            int(i[1]["x2"]),
            int(i[1]["y2"]),
        )
        mid = ((x1 + x2) / 2) + ((y1 + y2) / 2) * 1080
        area = (x2 - x1) * (y2 - y1)
        curr[i[0]].append(
            {
                "mid": mid,
                "area": area,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "dir": None,
            }
        )
    return curr


def movement(prev, curr):
    for i in curr:
        if i in prev:
            p = sorted(prev[i], key=lambda x: x["mid"])
            c = sorted(curr[i], key=lambda x: x["mid"])
            for objp, objc in zip(p, c):
                curr_area = objc["area"]
                prev_area = objp["area"]
                if curr_area < prev_area:
                    objc["dir"] = "Moving Away"
                else:
                    objc["dir"] = "Moving Towards"


def reader(frame):
    result = ocr.ocr(frame, cls=True)
    text = ""
    if result[0] is not None:
        for i in result[0]:
            _, t = i
            text += " " + t[0]
    return text


yolo = YOLO("yolov8n.pt")
ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

cap = cv2.VideoCapture("/home/shusrith/vids/l1.mp4")

frame_count = 0
prev, curr = {}, {}


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 15 != 0:
        continue
    text = reader(frame)
    results = yolo(frame)
    res = [(i["name"], i["box"]) for j in results for i in j.summary()]
    prev = curr
    curr = org(res)
    if prev is not None:
        movement(prev, curr, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
