import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#points = [[[] for i in range(68)]]]


def pointCoords(frame):
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    landmarks = predictor(grey, face)
    
    x = []
    y = []

    for n in range(0,68):
        x.append(landmarks.part(n).x)
        y.append(landmarks.part(n).y)

    maxy = max(y)

    y2 = []

    for val in y:
        y2.append(maxy-val)
        

    points = []

    for i in range(len(x)):
        points.append((x[i],y2[i]))
        
    return points
    
    
    
repeat = False

counter = 0

newPerson = []

while True:
    
    _, frame = cap.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(grey)
    for face in faces:

        landmarks = predictor(grey, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
    frame = cv2.flip(frame,1)
    cv2.putText(frame,
                'Press \'n\' to add a new person',
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (50,230,0),
                2,
                cv2.LINE_AA)

    cv2.imshow("Frame",frame)

    key = cv2.waitKey(1)
    if key == 27:
        cv2.destroyAllWindows()
        break
    elif key == ord('n') or repeat:
        if counter < 4:
            
            repeat = True
            counter += 1
            
            newPerson.append(pointCoords(frame))
            
        else:
            repeat = False
            counter = 0
            
            
        

