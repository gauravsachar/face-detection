import opencv
import cv2
import time
import numpy as np


vid=cv2.VideoCapture(0)

hand_cascade= cv2.CascadeClassifier('haarcascades/haarcascade_hand.xml')
while (True):
    ret, frame=vid.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    hand=hand_cascade.detectMultiScale (gray,1.1,4)
    
    for (a,b,c,d) in hand :
        cv2.rectangle(frame,(a,b),(a+c,b+d), (25,69,0),10)

    cv2.imshow('gaurav webcam hand gesture',frame)
    cv2.waitKey(1)
    
##    detections=cascade.detectMultiScale(frame, scaleFactor=1.3,minNeighbors=5)
##    #detections = cascade_classifier.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
##
##    if(len(detections) > 0):
##        (x,y,w,h) = detections[0]
##        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
##
    # for (x,y,w,h) in detections:
    # 	frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
##    
vid.release()
cv2.destroyAllWindows()
##
