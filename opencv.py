import opencv
import cv2


##img = cv2.imread('vegitto.png')
##
##cv2.imshow('image of vegitto',img)
##
##cv2.waitKey()
##

vid=cv2.VideoCapture(0)
face_cascade= cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades\haarcascade_eye.xml')


while (True):
    ret, frame=vid.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale (gray,1.1,4)

    for (a,b,c,d) in faces :
        cv2.rectangle(frame,(a,b),(a+c,b+d), (25,69,0),10)
        roi_gray = gray[b:b+d, a:c+d]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        roi_color = frame[b:b+d, a:a+c]
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        

    
    

    cv2.imshow('gaurav webcam',frame)
    cv2.waitKey(1)
    if (len(eyes)>5):
        break   

##    detections=cascade.detectMultiScale(frame, scaleFactor=1.3,minNeighbors=5)
##    #detections = cascade_classifier.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
##
##    if(len(detections) > 0):
##        (x,y,w,h) = detections[0]
##        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
##
    # for (x,y,w,h) in detections:
    # 	frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    



##    if cv2.waitKey(1) & 0xFF == ord('q'):
##        break
##    
vid.release()
cv2.destroyAllWindows()
##
