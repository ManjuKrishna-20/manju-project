import cv2
import numpy as np
faceDetect= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("trainingData.yml")
id=0
font= cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces =faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(128,128,128),3)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        print (rec.predict(gray[y:y+h,x:x+w]))
        if (id==1 and conf<60):
            id="Muruganandham"
            #cv2.namedWindow("id",cv2.WINDOW_NORMAL)
            #a=input("enter the otp: ")
            #cv2.imshow("",gray)
        elif (id==2 and conf<60):
            id="Madhan"
        elif (id==3 and conf<60):
            id="Sudhan"
        else:
            id="un_known"
                 
        
            
           
        cv2.putText(img,id, (x,y-10), font,0.55,(0,255,0),1) 
    cv2.imshow("Face",img);
    if(cv2.waitKey(5)==  ord('q')):
        break
    
cam.release()
cv2.destroyAllWindows()
        
    
