from datetime import datetime

from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
import numpy as np
import face_recognition
import cv2
import os
def capture(request):
    path = 'images'
    images = []
    classNames = []
    myList = os.listdir(path)

    for cl in myList:
        currentImg = cv2.imread(f'{path}/{cl}')
        images.append(currentImg)
        classNames.append(os.path.splitext(cl)[0])

    def findEncodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            encodeImg = face_recognition.face_encodings(img)[0]
            encodeList.append(encodeImg)
        return encodeList
    def markAttendance(name):
        with open('./mini/attendance.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
                if name not in nameList:
                    now = datetime.now()
                    dtString = now.strftime("%H:%M:%S")
                    f.writelines(f'\n{name}, {dtString}')
                    break

    encodeListsKnown = findEncodings(images)
    print(f' Total No of encodings :{len(encodeListsKnown[0])}')
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_RGB2BGR)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        if len(facesCurFrame) == 0:
            cv2.putText(img, "No Face found ", (100, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
        for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListsKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListsKnown, encodeFace)
            matchIndex = np.argmin(faceDis)
            if (matches[matchIndex]):
                name = classNames[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, "Attendance marked for "+name, (x1-200, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
                markAttendance(name)
            else:
                cv2.putText(img, "No matches found ", (x1 - 200, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return render(request, 'index.html')


def index(request):
    return render(request, 'index.html')


