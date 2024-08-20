import cv2
from ultralytics import YOLO
import logging

logging.getLogger("ultralytics").setLevel(logging.ERROR)


def org(res, curr):
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
                {"mid": mid, "area": area, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
            )
        else:
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
                }
            )
    return curr


def movement(prev, curr, frame):
    for i in curr:
        if i in prev:
            prev = sorted(prev[i], key=lambda x: x["mid"])
            curr = sorted(curr[i], key=lambda x: x["mid"])
            for objp, objc in zip(prev, curr):
                curr_area = objc["area"]
                prev_area = objp["area"]
                print(curr_area, prev_area)
                if curr_area < prev_area:
                    label = "Moving Away"
                else:
                    label = "Moving Towards"

                x1, y1, x2, y2 = (
                    objc["x1"],
                    objc["y1"],
                    objc["x2"],
                    objc["y2"],
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )


yolo = YOLO("yolov10x.pt")
cap = cv2.VideoCapture("/home/shusrith/vids/l4.mp4")
frame_count = 0
prev, curr = {}, {}
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 15 != 0:
        continue

    results = yolo(frame)
    res = [(i["name"], i["box"]) for j in results for i in j.summary()]
    prev = curr
    curr = org(res, curr)
    if prev is not None:
        movement(prev, curr, frame)

    # for match in res:
    #     cv2.rectangle(
    #         frame,
    #         (int(match[1]["x1"]), int(match[1]["y1"])),
    #         (int(match[1]["x2"]), int(match[1]["y2"])),
    #         (0, 255, 0),
    #         2,
    #     )
    #     cv2.putText(
    #         frame,
    #         f"{match[0]}",
    #         (int(match[1]["x1"]), int(match[1]["y1"])),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.5,
    #         (0, 255, 0),
    #         2,
    #     )
    curr = {}
    cv2.imshow("frame1", frame)
    if cv2.waitKey(1000) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
