import cv2
from ultralytics import YOLO
import logging

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
                if curr_area < prev_area:
                    label = "Moving Away"
                else:
                    label = "Moving Towards"
                print(i, objc["mid"], label)
                cv2.rectangle(
                    frame,
                    (
                        objc["x1"],
                        objc["y1"],
                    ),
                    (objc["x2"], objc["y2"]),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    label,
                    (objc["x1"], objc["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )


# Start tracing memory allocations
# tracemalloc.start()

yolo = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(r"C:\Users\swath\Downloads\blind-eyes\vids\l1.mp4")

frame_count = 0
prev, curr = {}, {}

# # Take a snapshot before processing
# snapshot1 = tracemalloc.take_snapshot()

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
    curr = org(res)
    if prev is not None:
        movement(prev, curr, frame)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1500) & 0xFF == ord("q"):
        break

# Take a snapshot after processing
# snapshot2 = tracemalloc.take_snapshot()

cap.release()
cv2.destroyAllWindows()

# # Compare snapshots
# top_stats = snapshot2.compare_to(snapshot1, "lineno")

# print("[ Top 10 differences ]")
# for stat in top_stats[:10]:
#     print(stat)

# # Get current and peak memory usage
# current, peak = tracemalloc.get_traced_memory()
# print(f"Current memory usage: {current / 1024 / 1024:.2f} MiB")
# print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MiB")

# # Stop tracing memory allocations
# tracemalloc.stop()
