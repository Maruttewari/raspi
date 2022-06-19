import cv2
import time
import datetime as dt
import os
names = ['None', 'Marut', 'Ajay', 'Praween', 'Suvishi']
waiting_seconds = 0.05 #sec for which camera will wait before taking next image
#                     so that single person is not entered multiple times
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
# indicate id counter
id = 0
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)   #camera for  entry gate (IN)
cam2=cv2.VideoCapture(1) #camera for exit gate (OUT)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height
# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
today = dt.datetime.now().strftime("%d_%m_%y")
while True:
    if os.path.exists("records/record_"+today+".csv"):
        file=open("records/record_"+today+".csv","a")
    else:
        file=open("records/record_"+today+".csv","w")
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    name=names[0]
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        # If confidence is less them 100 ==> "0" : perfect match
        if (confidence < 100):
            name = names[id]
        else:
            name = "unknown"

    #cv2.imshow('camera', img)
    #print(id)
    if name is not "None":
        file.write(str(id)+',')
        file.write(name+',')
        file.write(str(dt.date.today())+',')
        file.write(dt.datetime.now().strftime("%H:%M:%S")+',')
        file.write('IN')
        file.write('\n')
    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
    time.sleep(waiting_seconds)
    file.close()
    file = open("records.csv", "a")
    ret, img = cam2.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    name = names[0]
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        # If confidence is less them 100 ==> "0" : perfect match
        if (confidence < 100):
            name = names[id]
        else:
            name = "unknown"

    # cv2.imshow('camera', img)
    # print(id)
    if name is not "None":
        file.write(str(id) + ',')
        file.write(name + ',')
        file.write(str(dt.date.today()) + ',')
        file.write(dt.datetime.now().strftime("%H:%M:%S")+',')
        file.write('OUT')
        file.write('\n')
    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
    time.sleep(waiting_seconds)
    file.close()
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cam2.release()
cv2.destroyAllWindows()