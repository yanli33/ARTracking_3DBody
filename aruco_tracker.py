"""
Framework   : OpenCV Aruco
Description : Calibration of camera and using that for finding pose of multiple markers
Status      : Working
References  :
    1) https://docs.opencv.org/3.4.0/d5/dae/tutorial_aruco_detection.html
    2) https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
    3) https://docs.opencv.org/3.1.0/d5/dae/tutorial_aruco_detection.html
"""

import numpy as np
import cv2
from cv2 import aruco as aruco

import cv2

# 读取相机内参
cv_file = cv2.FileStorage("charuco_camera_calibration.yaml", cv2.FILE_STORAGE_READ)
camera_matrix= cv_file.getNode("camera_matrix").mat()
dist_matrix = cv_file.getNode("dist_coeff").mat()
cv_file.release()

# 设置marker相关参数
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
parameters = aruco.DetectorParameters_create()
parameters.adaptiveThreshConstant = 10

f_out = 'data.txt'
# 打开相机
cap = cv2.VideoCapture(0)
###------------------ ARUCO TRACKER ---------------------------
while (cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. 检测标记
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if np.all(ids != None):

        # 2. 获取标记的r和t向量
        rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_matrix )
        
        for i in range(0, ids.size):
            cv2.drawFrameAxes(frame, camera_matrix, dist_matrix , rvec[i], tvec[i], 0.1)
        # 3. 得到欧拉角并存到输出文件
        rotMat,_ = cv2.Rodrigues(rvec[0][0])
        projMat = np.hstack((rotMat, tvec[0][0].reshape(3, 1)))
        eulerAngles = cv2.decomposeProjectionMatrix(projMat)[6]
        print("eular:  " ,eulerAngles.reshape(1,3))

        try:
            np.savetxt(f_out, eulerAngles, fmt='%.2f')
        except IOError as e:
            print("文件写入错误:", e)
        aruco.drawDetectedMarkers(frame, corners)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


