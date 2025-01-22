import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText

# Load Images and Class Names
path = 'Images_Attendance'
images = [cv2.imread(f'{path}/{img}') for img in os.listdir(path)]
classNames = [os.path.splitext(img)[0].strip().upper() for img in os.listdir(path)]

# Find Encodings
def findEncodings(images):
    return [face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[0] for img in images]

# Mark Attendance
def markAttendance(name):
    with open('Attendance.csv', 'a+') as f:
        f.seek(0)
        attendanceList = [line.split(',')[0].strip().upper() for line in f.readlines()]
        if name not in attendanceList:
            time_now = datetime.now().strftime('%H:%M:%S')
            date_now = datetime.now().strftime('%d/%m/%Y')
            f.write(f'{name},{time_now},{date_now}\n')

# Get Absent Students
def getAbsentStudents():
    with open('Attendance.csv', 'r') as f:
        markedNames = [line.split(',')[0].strip().upper() for line in f.readlines()]
    absentStudents = [name for name in classNames if name not in markedNames]
    with open('Absent_Students.csv', 'w') as f:
        f.write('Absent Students\n')
        f.writelines(f'{student}\n' for student in absentStudents)
    return absentStudents

# Send Absent Students via Email
def sendEmail(absentStudents,recipient_email):
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

# Main Code
encodeListKnown = findEncodings(images)
print('Encoding Complete')

start_time = input("Enter start time (HH:MM:SS): ")
end_time = input("Enter end time (HH:MM:SS): ")
start_time = datetime.strptime(start_time, '%H:%M:%S').time()
end_time = datetime.strptime(end_time, '%H:%M:%S').time()

cap = cv2.VideoCapture(0)
while True:
    current_time = datetime.now().time()

    if current_time < start_time:
        print("Attendance has not started yet.")
        cv2.waitKey(1000)
        continue

    if current_time > end_time:
        print("Attendance window is closed.")
        break

    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex]
            y1, x2, y2, x1 = [v * 4 for v in faceLoc]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(10) == 13:
        break

cap.release()
cv2.destroyAllWindows()

absentStudents = getAbsentStudents()
sendAbsentStudentsEmail(absentStudents, 'sadongaredata@example.com')