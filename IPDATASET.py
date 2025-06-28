# import cv2
# import numpy as np
# import urllib.request as urllib

# url="http://192.168.1.57:8080/shot.jpg"
# def raw():
    
    
#     faceDetect=cv2.CascadeClassifier("C:/Users/Admin/Downloads/Code Correct/Code Correct/haarcascade_frontalface_default.xml")
   


import cv2
import numpy as np
import urllib.request as urllib
import os

url = "http://192.0.0.4:8080/shot.jpg"
faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def raw():
    name = input("Enter your name: ")
    
    # Assign a unique ID based on name (or load from file)
    if not os.path.exists("users.txt"):
        with open("users.txt", "w") as f:
            pass

    with open("users.txt", "r") as f:
        lines = f.readlines()
        names = [line.strip().split(",")[1] for line in lines]
        ids = [int(line.strip().split(",")[0]) for line in lines]
    
    if name in names:
        id = ids[names.index(name)]
    else:
        id = max(ids) + 1 if ids else 1
        with open("users.txt", "a") as f:
            f.write(f"{id},{name}\n")

    num = 0
    while True:
        imgPath = urllib.urlopen(url)
        imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            num += 1
            cv2.imwrite(f"database/Person.{id}.{num}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x + w, y + h), (211, 211, 211), 2)
            cv2.waitKey(100)

        cv2.imshow("face", img)
        if cv2.waitKey(1) == ord('q') or num >= 10:
            break

    cv2.destroyAllWindows()

while True:
    raw()

#     num=0

#     while True:
#         imgPath=urllib.urlopen(url)
#         imgNp=np.array(bytearray(imgPath.read()),dtype=np.uint8)
#         img=cv2.imdecode(imgNp,-1)
#         gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         face=faceDetect.detectMultiScale(gray,1.3,5)
#         for (x,y,w,h) in face:
            
#             num=num+1
#             cv2.imwrite("database/Person."+str(id)+"."+str(num)+".jpg",gray[y:y+h,x:x+h])
#             cv2.rectangle(img, (x,y), (x+w,y+h), (211,211,211), 2)
#             cv2.waitKey(100)
            
            
#         cv2.imshow("face",img)
#         cv2.waitKey(1)
#         if (num>10):
#             break
    
        
  
#     cv2.destroyAllWindows()

# while True:
#     raw()

    
