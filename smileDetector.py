import cv2

trained_face_data = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
trained_smile_data = cv2.CascadeClassifier('./haarcascade_smile_default.xml')
webcam = cv2.VideoCapture(0)

while (True):

    sucess, frame = webcam.read() 
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5, minSize=(45, 45))# 1.3, 5)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)
        the_face = frame[y: y+h, x: x+w]
        face_gray = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        smiles = trained_smile_data.detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=3, minSize=(15, 15))
        if len(smiles) > 0:
            for (x_, y_, w_, h_) in smiles:
                cv2.rectangle(the_face, (x_,y_), (x_ + w_, y_ + h_), (0, 255, 0), 2)
                cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale = 3, fontFace = cv2.FONT_HERSHEY_PLAIN, color = (255, 0, 0))
        else:
            cv2.putText(frame, 'Why so serious?', (x, y+h+40), fontScale = 3, fontFace = cv2.FONT_HERSHEY_PLAIN, color = (255, 0, 0))


    cv2.imshow('face detected', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

print("Done")