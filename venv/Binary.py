import cv2
from PIL import Image
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

def BRZ(im):
    threshold = 150
    imm = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    image = np.array(imm)

    for i in range(len(image)):
        for j in range(len(image[0])):
                if image[i, j] > threshold:
                    image[i, j] = 255
                else:
                    image[i, j] = 0

    cv2.imshow("OTSU", image)
    return image


def Morphological_transformations(image, type):
    kernel = np.ones((5, 5), np.uint8)

    result = {
        1: cv2.erode(image, kernel, iterations=1),
        2: cv2.dilate(image, kernel, iterations=1),
        3: cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel),
        4: cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel),
        5: cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel),
        6: cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel),
        7: cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    }
    default_value = -1
    cv2.imshow("erosion", result.get(type, default_value))
