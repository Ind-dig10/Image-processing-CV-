import cv2 
import numpy as np

img = cv2.imread('Result.jpg',0)
imgc = cv2.imread('Result.jpg')
mark = np.zeros(img.shape[:2], dtype = np.int32)
mark[200, 190] = 1
mark[200,200] = 2
mark[1,1] = 5


water = cv2.watershed(imgc, mark)

water *= 60
water = water.astype(np.uint8)
cv2.circle(water, (350,350), 5, (255,255,0) )
cv2.circle(water, (230,200), 5, (255,255,0) )
cv2.circle(water, (1,1), 5, (255,255,0) )
cv2.imshow('threshold', imgc)

cv2.imshow('water', water)
while(True):
    if cv2.waitKey(33) == ord('q'):
        break
cv2.destroyAllWindows()
