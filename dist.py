import time
import cv2
from ultralytics import YOLO
import logging
import math

logging.getLogger("ultralytics").setLevel(logging.ERROR)


yolo = YOLO("yolov10x.pt")
cap = cv2.VideoCapture("/home/shusrith/vids/l4.mp4")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 15 != 0:
        continue

    results = yolo(frame)
    res = [(i["name"], i["box"]) for j in results for i in j.summary()]
    print("Res", res)
    for match in res:
        cv2.rectangle(
            frame,
            (int(match[1]["x1"]), int(match[1]["y1"])),
            (int(match[1]["x2"]), int(match[1]["y2"])),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"{match[0]}",
            (int(match[1]["x1"]), int(match[1]["y1"])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.imshow("frame1", frame)
    if cv2.waitKey(1000) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
