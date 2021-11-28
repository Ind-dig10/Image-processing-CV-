import cv2
import numpy as np

#img = cv2.imread('apple.jpg',0)
imgc =  cv2.imread('lion.jpg')
img = imgc.reshape((-1,3))
img = np.float32(img)

r, l, c = cv2.kmeans(img,4,None, (cv2.TERM_CRITERIA_EPS+ cv2.TERM_CRITERIA_MAX_ITER, 10,1.0),5, cv2.KMEANS_RANDOM_CENTERS )

c = np.uint8(c)
r = c[l.flatten()]
r2 = r.reshape((imgc.shape))


cv2.imshow('image',r2)
while(True):
    if cv2.waitKey(33) == ord('q'):
        break
cv2.destroyAllWindows()
0