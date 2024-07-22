import time
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import math

logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Load YOLO model
yolo = YOLO("yolov8n.pt")

# Open video capture for both cameras
cap = cv2.VideoCapture("/home/shusrith/vids/l4.mp4")
cap1 = cv2.VideoCapture("/home/shusrith/vids/r4.mp4")
frame_count = 0

# Load calibration data
calib_data = np.load("calib.npz")
mtxL = calib_data["mtxL"]
distL = calib_data["distL"]
mtxR = calib_data["mtxR"]
distR = calib_data["distR"]
R = calib_data["R"]
T = calib_data["T"]

# Create SIFT detector
sift = cv2.SIFT_create()

while True:
    ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    if not ret or not ret1:
        break
    frame_count += 1
    if frame_count % 15 != 0:
        continue

    # YOLO object detection
    results = yolo(frame)
    results1 = yolo(frame1)
    res = [(i["name"], i["box"]) for j in results for i in j.summary()]
    res1 = [(i["name"], i["box"]) for j in results1 for i in j.summary()]

    if res == [] or res1 == []:
        continue
    for objL, objR in zip(res, res1):
        nameL, boxL = objL
        nameR, boxR = objR

        # Ensure that the same object is detected in both frames
        if nameL != nameR:
            continue
        try:
            x1L, y1L, x2L, y2L = map(int, boxL.values())
            x1R, y1R, x2R, y2R = map(int, boxR.values())
        except:
            print(boxL, boxR)
            break

        roiL = frame[y1L:y2L, x1L:x2L]
        roiR = frame1[y1R:y2R, x1R:x2R]

        kpL, desL = sift.detectAndCompute(roiL, None)
        kpR, desR = sift.detectAndCompute(roiR, None)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(desL, desR)
        matches = sorted(matches, key=lambda x: x.distance)

        disparities = []
        for match in matches:
            ptL = kpL[match.queryIdx].pt
            ptR = kpR[match.trainIdx].pt
            disparity = ptL[0] - ptR[0]
            disparities.append(disparity)

        if len(disparities) > 0:
            B = np.linalg.norm(T)
            f = mtxL[0, 0]
            Z_values = [(f * B) / d for d in disparities if d > 0]
            object_depth = np.mean(Z_values) if Z_values else None

            if object_depth is not None:
                cv2.rectangle(frame, (x1L, y1L), (x2L, y2L), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{nameL}: {object_depth:.2f} units",
                    (x1L, y1L - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                cv2.rectangle(frame1, (x1R, y1R), (x2R, y2R), (0, 255, 0), 2)
                cv2.putText(
                    frame1,
                    f"{nameR}: {object_depth:.2f} units",
                    (x1R, y1R - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
            else:
                print(f"Could not estimate depth for {nameL}")
        else:
            print(f"No valid disparities found for {nameL}")

    # Display the frames with annotations
    combined_frame = np.hstack((frame, frame1))
    cv2.imshow("Stereo Camera View", combined_frame)

    if cv2.waitKey(1000) & 0xFF == ord("q"):
        break

cap.release()
cap1.release()
cv2.destroyAllWindows()
