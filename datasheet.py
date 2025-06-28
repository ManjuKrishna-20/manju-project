import cv2
import numpy as np

def raw():
    
    faceDetect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cam=cv2.VideoCapture(0)


    id=int(input("enter your id:"))
    num=0

    while True:
        ret,img = cam.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face=faceDetect.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face:
            
            num=num+1
            cv2.imwrite("database/Person."+str(id)+"."+str(num)+".jpg",gray[y:y+h,x:x+h])
            cv2.rectangle(img, (x,y), (x+w,y+h), (211,211,211), 2)
            cv2.waitKey(100)
            
            
        cv2.imshow("face",img)
        cv2.waitKey(1)
        if (num>5):
            break
    
        
    cam.release()
    cv2.destroyAllWindows()

while True:
    raw()

    
