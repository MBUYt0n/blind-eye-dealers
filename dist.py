import time
import cv2
from ultralytics import YOLO
import logging
import math

logging.getLogger("ultralytics").setLevel(logging.ERROR)

B = 0.751
F = 550
W = 540
FOV = math.radians(80)


def calculate_iou(box1, box2):
    xA = max(box1["x1"], box2["x1"])
    yA = max(box1["y1"], box2["y1"])
    xB = min(box1["x2"], box2["x2"])
    yB = min(box1["y2"], box2["y2"])
    interArea = abs(xB - xA) * abs(yB - yA)
    box1Area = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    box2Area = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
    unionArea = box1Area + box2Area - interArea
    iou = interArea / unionArea
    return iou


def find_matches(res, res1, iou_threshold=0.1):
    matches = []
    for obj1 in res:
        best_match = None
        best_iou = iou_threshold
        for obj2 in res1:
            if obj1[0] == obj2[0]:
                iou = calculate_iou(obj1[1], obj2[1])
                if iou > best_iou:
                    best_iou = iou
                    best_match = obj2
        if best_match:
            matches.append((obj1, best_match))
    return matches


# def calc_distance(param):
#     c = []
#     for i in param:
#         x, _, z, _ = i[1].values()
#         c.append(abs((x + z) / 2))
#     dist = (B * W * 2) / (2 * math.tan(FOV / 2) * abs(c[0] - c[1]))
#     return dist

def calc_distance(param):
    c = []
    for i in param:
        x, _, z, _ = i[1].values()
        c.append(abs(x + z))
    d = abs(c[0] - c[1])
    dist = (B * F) / d
    return dist


yolo = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("/home/shusrith/vids/l1.mp4")
cap1 = cv2.VideoCapture("/home/shusrith/vids/r1.mp4")
frame_count = 0

while True:
    ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    if not ret or not ret1:
        break
    frame_count += 1
    if frame_count % 15 != 0:
        continue

    results = yolo(frame)
    results1 = yolo(frame1)
    res = [(i["name"], i["box"]) for j in results for i in j.summary()]
    res1 = [(i["name"], i["box"]) for j in results1 for i in j.summary()]
    # print("Res", res)
    # print("Res1", res1)
    matches = find_matches(res, res1)
    for match in matches:
        d = calc_distance(match)
        print(f"Distance of {match[0][0]} : {d}m")
        cv2.rectangle(
            frame,
            (int(match[0][1]["x1"]), int(match[0][1]["y1"])),
            (int(match[0][1]["x2"]), int(match[0][1]["y2"])),
            (0, 255, 0),
            2,
        )
        cv2.rectangle(
            frame1,
            (int(match[1][1]["x1"]), int(match[1][1]["y1"])),
            (int(match[1][1]["x2"]), int(match[1][1]["y2"])),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"{d:.2f}m",
            (int(match[0][1]["x1"]), int(match[0][1]["y1"])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame1,
            f"{d:.2f}m",
            (int(match[1][1]["x1"]), int(match[1][1]["y1"])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    cv2.imshow("frame1", frame)
    cv2.imshow("frame2", frame1)
    if cv2.waitKey(1000) & 0xFF == ord("q"):
        break

cap.release()
cap1.release()
cv2.destroyAllWindows()
