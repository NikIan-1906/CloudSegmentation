import cv2
import numpy as np
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

def find_clouds(image):
    return cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), np.array([0, 0, 150]), np.array([179, 70, 255]))

while cap.isOpened():
    ret, frame = cap.read()

    if cv2.waitKey(33) & 0xFF == 32:
        img = frame
        #cv2.imshow("img", img)
        mask = find_clouds(img)
        cv2.imshow("clouds", mask)

    if cv2.waitKey(33) & 0xFF == ord("q") or not ret:
        break

    cv2.imshow("frame", frame)

cap.release()
cv2.destroyAllWindows()
