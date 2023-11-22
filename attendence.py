import face_recognition
import numpy as np
import os
import csv
import cv2
from datetime import datetime



video_capture = cv2.VideoCapture(0)

ijaz_image = face_recognition.load_image_file("photos/ijaz.jpg")
ijaz_encoding = face_recognition.face_encodings(ijaz_image)[0]

shaique_image = face_recognition.load_image_file("photos/shaique.jpg")
shaique_encoding = face_recognition.face_encodings(shaique_image)[0]

me_image = face_recognition.load_image_file("photos/me.jpg")
me_encoding = face_recognition.face_encodings(me_image)[0]

joydeep_image = face_recognition.load_image_file("photos/joydeep.jpg")
joydeep_encoding = face_recognition.face_encodings(joydeep_image)[0]

known_face_encoding = [
    ijaz_encoding,
    shaique_encoding,
    me_encoding,
    joydeep_encoding
    
]

known_faces = [
    "ijaz",
    "shaique",
    "me",
    "joydeep"
]

students = known_faces.copy()

face_locations = []
face_encodings = []
face_names =[]
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv','w+',newline='')
fwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgbsmall_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgbsmall_frame)
        face_encodings = face_recognition.face_encodings(rgbsmall_frame,face_locations)
        face_names=[]
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            match_index = np.argmin(face_distance)
            if matches[match_index]:
                name = known_faces[match_index]

            face_names.append(name)
            if name in known_faces:
                if name in students:
                    students.remove(name)
                    print("Students left to give attendence",students)
                    current_time= now.strftime("%H-%M-%S")
                    fwriter.writerow([name,current_time])
            else:
                print("Student not found!")
                break
    
    cv2.imshow("Attendence Monitor",frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
video_capture.release()
cv2.destroyAllWindows()
f.close()


