import cv2
import numpy as np
import matplotlib.pyplot as plt



img1 = cv2.imread('2.jpg')
# or use cv2.CV_LOAD_IMAGE_GRAYSCALE
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
cv2.imshow('input', img1)
w,h = img1.shape
# make a 32bit float for doing the dct within
img2 = np.zeros((w,h), dtype=np.float32)

img2 = img2+img1[:w, :h]
dct1 = cv2.dct(img2)


recor_temp = dct1[0:100,0:100]
recor_temp2 = np.zeros(img1.shape)
recor_temp2[0:100,0:100] = recor_temp
img_recor1 = cv2.idct(recor_temp2)

 #Compressed picture recovery

key = -1
while(key < 0):
    cv2.imshow("DCT", dct1)
    cv2.imshow("result", img_recor1)
    key = cv2.waitKey(1)
cv2.destroyAllWindows()