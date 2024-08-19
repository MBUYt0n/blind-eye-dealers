import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_disparity(left_img, right_img):
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=16 * 10, blockSize=15)
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    return disparity


cap = cv2.VideoCapture("/home/shusrith/vids/l1.mp4")
cap1 = cv2.VideoCapture("/home/shusrith/vids/r1.mp4")
while True:
    ret, left_img = cap.read()
    ret1, right_img = cap1.read()
    if not ret or not ret1:
        break
    disparity_map = compute_disparity(left_img, right_img)
    cv2.imshow("Disparity Map", disparity_map)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
