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
                {0: {"mid": mid, "area": area, "x1": x1, "y1": y1, "x2": x2, "y2": y2}}
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
                    list(curr[i[0]][-1].keys())[0]
                    + 1: {
                        "mid": mid,
                        "area": area,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                }
            )
    return curr


def matcher(prev, curr):
    combined = []
    for d in prev:
        combined.append(d)
    for d in curr:
        combined.append(d)
    sorted_combined = sorted(combined, key=lambda x: list(x.values())[0]["mid"])
    return sorted_combined


def movement(prev, curr, frame):
    for i in curr:
        if i in prev:
            sorted_combined = matcher(prev[i], curr[i])
            for obj in sorted_combined:
                obj_id = list(obj.keys())[0]
                curr_obj = obj[obj_id]
                try:
                    print("1", prev[i][1])
                except:
                    print("0", prev[i][0])
                prev_obj = prev[i][0][obj_id]
                curr_area = curr_obj["mid"]
                prev_area = prev_obj["mid"]

                if curr_area < prev_area:
                    label = "Moving Away"
                else:
                    label = "Moving Towards"

                x1, y1, x2, y2 = (
                    curr_obj["x1"],
                    curr_obj["y1"],
                    curr_obj["x2"],
                    curr_obj["y2"],
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
    curr = {}
    cv2.imshow("frame1", frame)
    if cv2.waitKey(1000) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
