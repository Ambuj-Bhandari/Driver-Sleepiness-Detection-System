import os
import dlib
import numpy as np
import utils
import cv2
from imutils import face_utils

video=cv2.VidepCapture(0)

detect=dlib.get_frontal_face_detector()
predict=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


while(True):
    frame=video.read()
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    face_feature=detect(frame_gray)
    
    for i in face_feature:
        x1=i.left()
        y1=i.top()
        x2=i.right()
        y2=i.bottom()
        
    face_frame=frame.copy()
    face=cv2.rectangle(face_frame,(x1,y1),(x2,y2),(0,255,0),2)
    
    pinpoints=predict(frame_gray,face)
    pinpoints=face_utils.shape_to_np(pinpoints)
    
    left_eye=compute_eyes(pinpoints[36],pinpoints[37],pinpoints[38],pinpoints[41],pinpoints[40],pinpoints[39])
    right_eye= compute_eyes(pinpoints[42],pinpoints[43],pinpoints[44],pinpoints[47],pinpoints[46],pinpoints[45])
    reye=pinpoints[42:48]
    leye=pinpoints[36:42]
    
    lips=compute_lips(pinpoints)
    lip=pinpoints[48:60]
    
    cv2.drawContours(frame, [leye], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [reye], -1, (0, 255, 0), 1)
    
    
    
    
    
    

    
    
    