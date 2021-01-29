import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv

MODEL = 'hog'
TOLERANCE = 0.8
csvFileName=input("Enter the name of csv file : ")
row_list = [["ROLL NUMBER", "TIME", "ATTENDANCE"]]
path = 'DATASETOFIMAGES'
imagesData = []
classNames = []
myList = os.listdir(path)
#print(myList)
with open(csvFileName+'.csv','w',newline='') as file:
    writer = csv.writer(file)
    writer.writerows(row_list)
    print("CSV File Created")
for element in myList:
    curImg = cv2.imread(f'{path}/{element}')
    imagesData.append(curImg)
    classNames.append(os.path.splitext(element)[0])
#print(classNames)
 
def faceEncodings(imagesData): #function to encode the images 
    encodedList = []
    for img in imagesData:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodedList.append(encode)
    return encodedList
 
def markAttendance(name): #function to write roll no.s of present students

    with open(csvFileName+'.csv','r+') as newFile:
        myDataList = newFile.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            newFile.writelines(f'\n{name},{dtString},"PRESENT"')
            
encodeListKnown = faceEncodings(imagesData)
print("Encoding Complete")
 
cap = cv2.VideoCapture(0)
print("Opening Camera")

def faceDetection(): 
    while True: #loop to compare the webcam recorded face with dataset
        success, img = cap.read()
        imgS = cv2.resize(img,(0,0),None,0.17,0.17)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
     
        facesCurFrame = face_recognition.face_locations(imgS,model=MODEL)
        print(facesCurFrame)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
        print(encodesCurFrame)
     
        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace,TOLERANCE) #compare the encoded image to the webcam recorded image
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace) #calculate the face distance
            print(faceDis)
            matchIndex = np.argmin(faceDis)
            print(matchIndex)
            print(matches[matchIndex])  
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1,x2,y2,x1 = faceLoc 
                y1, x2, y2, x1 = y1*6,x2*6,y2*6,x1*6
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2) #putting a rectangular frame around face
                markAttendance(name) #sendind the recognized roll no. to function which writes attendance
     
        cv2.imshow('Webcam',img)
        if cv2.waitKey(1) & 0xFF == ord("c"):
            cv2.destroyAllWindows()
            break
            

faceDetection()
