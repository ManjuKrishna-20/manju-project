# import cv2
# import numpy as np
# import urllib.request as urllib

# faceDetect= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# #import urllib
# url="http://192.168.1.57:8080/shot.jpg"
# rec=cv2.face.LBPHFaceRecognizer_create()
# #rec=cv2.createLBPHFaceRecognizer()
# rec.read("trainingData.yml")
# id=0
# #font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,6,1,0,4)
# font= cv2.FONT_HERSHEY_SIMPLEX
# while True:
#     imgPath=urllib.urlopen(url)
#     imgNp=np.array(bytearray(imgPath.read()),dtype=np.uint8)
#     img=cv2.imdecode(imgNp,-1)
#     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     faces =faceDetect.detectMultiScale(gray,1.3,5)
#     for(x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(128,128,128),3)
#         id,conf=rec.predict(gray[y:y+h,x:x+w])
#         print (rec.predict(gray[y:y+h,x:x+w]))
#         if (id==1 and conf<60):
#             id="Muruganandham"
#             #cv2.namedWindow("id",cv2.WINDOW_NORMAL)
#             #a=input("enter the otp: ")
#             #cv2.imshow("",gray)
#         elif (id==2 and conf<60):
#             id="Madhan"
#         elif (id==3 and conf<60):
#             id="Sudhan"
#         else:
#             id="un_known"
           
#             #cv2.imshow("detected",gray)
#             #print ("alert")
            
#             #cv2.imwrite('lennapng.png',img)
#             #cv2.waitKey(200)
#         cv2.putText(img,id, (x,y-10), font,0.55,(0,255,0),1) 
#     cv2.imshow("Face",img);
#     if(cv2.waitKey(5)==  ord('q')):
#         break
    
# cam.release()
# cv2.destroyAllWindows()
        
    

import cv2
import numpy as np
import urllib.request as urllib

# Load face detector
faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load face recognizer model
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("trainingData.yml")  # Make sure you have this after training

# Load ID-name mapping from users.txt
id_name_map = {}
with open("users.txt", "r") as f:
    for line in f:
        user_id, name = line.strip().split(",")
        id_name_map[int(user_id)] = name

# Webcam IP stream
url = "http://192.0.0.4:8080/shot.jpg"

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    imgPath = urllib.urlopen(url)
    imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (128, 128, 128), 2)
        id, conf = rec.predict(gray[y:y+h, x:x+w])

        # Get name from ID, check confidence
        if conf < 60:
            name = id_name_map.get(id, "Unknown")
        else:
            name = "Unknown"

        # Display name
        cv2.putText(img, str(name), (x, y - 10), font, 0.55, (0, 255, 0), 1)

    cv2.imshow("Face", img)

    # Exit on 'q'
    if cv2.waitKey(5) == ord('q'):
        break

cv2.destroyAllWindows()
