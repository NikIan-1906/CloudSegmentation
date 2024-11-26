import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
cap.set(cv2.CAP_PROP_FPS, 30)

while cap.isOpened():
    ret, frame = cap.read()

    if cv2.waitKey(33) & 0xFF == 32:
        img = frame
        cv2.imshow("img", img)

    if cv2.waitKey(33) & 0xFF == ord("q") or not ret:
        break

    cv2.imshow("frame", frame)


cap.release()
cv2.destroyAllWindows()