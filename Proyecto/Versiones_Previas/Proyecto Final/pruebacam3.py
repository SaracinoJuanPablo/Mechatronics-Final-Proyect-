import cv2

cv2.namedWindow("Camara")
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Camara", frame)
    if cv2.waitKey(50) >= 0:
        break

cap.release()
cv2.destroyAllWindows()
