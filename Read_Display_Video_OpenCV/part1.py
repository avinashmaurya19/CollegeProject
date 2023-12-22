import cv2

cap = cv2.VideoCapture("../Video/demo.mp4")

ret , frame = cap.read()

cv2.imshow("Frame",frame)
cv2.waitKey(0)