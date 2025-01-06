import cv2

#Load pre-trained face classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#Read the image (for image file)
image = cv2.imread("Pic.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

#Show the image with detected image
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
