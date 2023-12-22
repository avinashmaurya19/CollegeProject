import cv2

image = cv2.imread("../Images/image1.jpg")

resize_image = cv2.resize(image , (0 , 0) , fx= 0.4 , fy = 0.4 , interpolation=cv2.INTER_AREA)

cv2.imshow("image" , resize_image)

cv2.waitKey(0)