import time
import cv2
from ultralytics import YOLO


# Function to calculate IoU
def calculate_iou(box1, box2):
    # Adjusted to access values by 'x1', 'x2', 'y1', 'y2'
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


# Function to find matches based on IoU
def find_matches(res, res1, iou_threshold=0.1):
    matches = []
    for obj1 in res:
        best_match = None
        best_iou = iou_threshold
        for obj2 in res1:
            if obj1[0] == obj2[0]:  # Matching based on the same object class
                iou = calculate_iou(obj1[1], obj2[1])
                if iou > best_iou:
                    best_iou = iou
                    best_match = obj2
        if best_match:
            matches.append((obj1, best_match))
    return matches


# Initialize YOLO
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
    print("Res", res)
    print("Res1", res1)
    matches = find_matches(res, res1)
    for match in matches:
        print(f"Match found: {match[0]} with {match[1]}")

    cv2.imshow("frame1", frame)
    cv2.imshow("frame2", frame1)
    if cv2.waitKey(2000) & 0xFF == ord("q"):
        break

cap.release()
cap1.release()
cv2.destroyAllWindows()

# Res = [
#     ("motorcycle", {"x1": 637.30859, "y1": 428.97052, "x2": 693.8136, "y2": 498.54309}),
#     ("car", {"x1": 571.50128, "y1": 403.45056, "x2": 675.40851, "y2": 456.21954}),
# ]
# Res1 = [
#     ("car", {"x1": 542.99652, "y1": 505.854, "x2": 658.11853, "y2": 557.58679}),
#     ("person", {"x1": 241.08696, "y1": 462.22861, "x2": 274.08539, "y2": 542.72144}),
#     (
#         "motorcycle",
#         {"x1": 610.31818, "y1": 537.30042, "x2": 655.71405, "y2": 598.70874},
#     ),
# ]
# l = find_matches(Res, Res1)
# print(l)