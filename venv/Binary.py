import cv2
import numpy as np

def adap_binary(image):
    result = cv2.adaptiveThreshold(image, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY ,41,3)
    cv2.imshow("test binary", result)

def threshold_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("input_gray_image", gray)

    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("threshold：%s" % ret)
    cv.imshow("OTSU", binary)