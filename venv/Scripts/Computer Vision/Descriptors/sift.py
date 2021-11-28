import numpy as np
import cv2 as cv


def sift_descriptor():
    img = cv.imread('lion.jpg')
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv.drawKeypoints(gray,kp,img)
    cv.imwrite('sift_keypoints.jpg',img)

