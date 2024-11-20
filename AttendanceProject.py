import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

path = 'Images_Attendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0].strip().upper())  # Uppercase and strip spaces for consistency
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    if isAttendanceOpen():
        with open('Attendance.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0].strip().upper())  # Ensure names are uppercase and stripped
            if name not in nameList:
                time_now = datetime.now()
                tString = time_now.strftime('%H:%M:%S')
                dString = time_now.strftime('%d/%m/%Y')
                f.writelines(f'\n{name},{tString},{dString}')
    else:
        print(f"Attendance is closed. {name}'s attendance cannot be marked.")

def isAttendanceOpen():
    start_time = datetime.strptime("12:19:00", "%H:%M:%S").time()  # e.g., 9:00 AM
    end_time = datetime.strptime("12:20:00", "%H:%M:%S").time()    # e.g., 10:00 AM
    current_time = datetime.now().time()

    if start_time <= current_time <= end_time:
        return True
    else:
        return False

def isBeforeStartTime():
    start_time = datetime.strptime("12:19:00", "%H:%M:%S").time()
    current_time = datetime.now().time()
    return current_time < start_time

def getAbsentStudents():
    with open('Attendance.csv', 'r') as f:
        myDataList = f.readlines()
        markedNames = [line.split(',')[0].strip().upper() for line in myDataList]  # Ensure consistency
    absentStudents = [student for student in classNames if student not in markedNames]
    
    with open('Absent_Students.csv', 'w') as f:
        f.write('Absent Students\n')
        for student in absentStudents:
            f.write(f'{student}\n')
    
    return absentStudents

def sendAbsentStudentsEmail(absentStudents, recipient_email):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('sadongare7@gmail.com', 'arco jxzu gtzd rvwc')

    msg = MIMEMultipart()
    msg['From'] = 'sadongare7@gmail.com'
    msg['To'] = 'sadongaredata@gmail.com'
    msg['Subject'] = 'Absent Students List'

    body = 'The following students were absent:\n' + '\n'.join(absentStudents)
    msg.attach(MIMEText(body, 'plain'))

    text = msg.as_string()
    server.sendmail('sadongare7@gmail.com', 'sadongaredata@gmail.com', text)
    server.quit()

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    if isBeforeStartTime():
        print("Attendance has not started yet. Please wait for the start time.")
        cv2.waitKey(1000)
        continue

    if not isAttendanceOpen():
        print("Attendance window is closed. Closing the webcam.")
        break

    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 250, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
        
    cv2.imshow('webcam', img)

    if cv2.waitKey(10) == 13:
        break

cap.release()
cv2.destroyAllWindows()

absentStudents = getAbsentStudents()
sendAbsentStudentsEmail(absentStudents, 'sadongaredata@example.com')