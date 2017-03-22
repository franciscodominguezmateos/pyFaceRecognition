'''
Created on Mar 18, 2017

@author: Francisco Dominguez
'''
import cv2
#cap0 = cv2.VideoCapture("http://192.168.43.240:8080/video")
cap0 = cv2.VideoCapture(0)
# Now process all the images
basepath="/home/francisco/face_recognition"
i=0
while True:
    ret,img0=cap0.read()
    if not ret:
        print "No puedo capturar imagenes"
    else:
        cv2.imshow("Cam",img0)
    c=cv2.waitKey(1)
    if c!=-1:
        if c==115:
            cv2.imwrite(basepath+"/face%i.jpg"%i,img0)
            i+=1
        if c==27:
            break
        print i,c
