import cv2

cap = cv2.VideoCapture("../Video/traffic.mp4")

if(cap.isOpened()==False):
    print("unable to read the video")

while (cap.isOpened()):
    ret,frame = cap.read()
    if ret:
        resize_frame = cv2.resize(frame,(0,0),fx=1,fy=1,interpolation=cv2.INTER_AREA)
        cv2.imshow("Frame" , resize_frame)
        if(cv2.waitKey(1) & 0xFF==ord('1')):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()