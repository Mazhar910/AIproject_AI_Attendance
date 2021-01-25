import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv

csvFileName=input("Enter the name of csv file : ")
path = 'ImagesAttendance'
imagesData = []
classNames = []
myList = os.listdir(path)
print(myList)
with open(csvFileName+'.csv','w',newline='') as file:
    writer = csv.writer(file)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    imagesData.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
 
def findEncodings(imagesData): #function to encode the images 
    encodeList = []
    for img in imagesData:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 
def markAttendance(name): #function to write roll no.s of present students

    with open(csvFileName+'.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            
encodeListKnown = findEncodings(imagesData)
print("Encoding complete")
 
cap = cv2.VideoCapture(0)
 
while True: #loop to compare the webcam recorded face with dataset
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace) #compare the encoded image to the webcam recorded image
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace) #calculate the face distance
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2) #putting a rectangular frame around face
            markAttendance(name) #sendind the recognized roll no. to function which writes attendance
 
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
