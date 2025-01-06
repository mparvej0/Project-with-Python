import cv2

#Initialize webcam
cap = cv2.VideoCapture(0)

#Load pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    #Read each frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    #Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    #Draw rectagles aroud faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    #Display the resulting frame
    cv2.imshow('Face Detection', frame)

#Break the loop wh3n the "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

#Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
