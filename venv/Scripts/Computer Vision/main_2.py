from pylab import *
from PIL import Image
import Point_Detector as harris
import cv2 as cv
import numpy as np

image = cv.imread("kzn_3.jpg")
h, w = image.shape[:2]
cv.imshow("dsf", image)
print(h/2, w/2, end='\n')
print('a', 'b', 'c', sep='*')
print('d', 'e', 'f', sep='**', end='')
print('g', 'h', 'i', sep='+', end='%')
print('j', 'k', 'l', sep='-', end='\n')
cv.waitKey()