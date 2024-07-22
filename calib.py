import cv2
import numpy as np
import os
from PIL import Image
# Define the criteria for termination of the iterative process of corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
l = os.listdir("Checkers")
r = [i for i in l if i.startswith("r")]
l = [i for i in l if i.startswith("l")]
r = sorted(r)
l = sorted(l)
r = [np.array(Image.open(f"Checkers/{i}")) for i in r]
l = [np.array(Image.open(f"Checkers/{i}")) for i in l]

objpoints = []  # 3d point in real world space
imgpointsL = []  # 2d points in image plane.
imgpointsR = []  # 2d points in image plane.



for imgL, imgR in zip(l, r):
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv2.findChessboardCorners(grayL, (9, 6), None)
    retR, cornersR = cv2.findChessboardCorners(grayR, (9, 6), None)

    if retL and retR:
        objpoints.append(objp)

        cornersL2 = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsL.append(cornersL2)

        cornersR2 = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR2)

# Calibrate the stereo camera
ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpointsL,
    imgpointsR,
    grayL.shape[::-1],
    None,
    None,
    None,
    None,
    flags=cv2.CALIB_FIX_INTRINSIC,
)

np.savez("calib.npz", mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T, E=E, F=F)