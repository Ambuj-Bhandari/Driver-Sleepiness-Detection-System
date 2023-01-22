import os
import dlib
import numpy as np
import utils
import cv2

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
    cv2.rectangle(face_frame,(x1,y1),(x2,y2),(0,255,0),2)
    
    pinpoints=predict(frame_gray,face_feature)
    pinpoints=face_utils.shape_to