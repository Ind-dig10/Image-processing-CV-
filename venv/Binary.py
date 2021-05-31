import cv2
from PIL import Image
import numpy as np


def threshold_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("input_gray_image", gray)

    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("threshold：%s" % ret)
    cv.imshow("OTSU", binary)


def binary_image_transformations(im):
    threshold = 100
    imm = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    image = np.array(imm)

    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i, j] > threshold:
                image[i, j] = 255
            else:
                image[i, j] = 0

    return image


def Morphological_transformations(image, type):
    kernel = np.ones((3, 3), np.uint8)

    result = {
        1: cv2.erode(image, kernel, iterations=1),  # Эрозия
        2: cv2.dilate(image, kernel, iterations=1),  # Наращивание
        3: cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel),
        4: cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel),
        5: cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel), #Градиент
        6: cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel),
        7: cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    }
    default_value = -1
    res = result.get(type, default_value)
    return res

def erosion(image, binary_image):
    binary = binary_image
    erode = np.zeros(image.shape)

    for i in range(0, 400):
        for j in range(0, 400):
            if i != 0 and j != 0:
                erode[i, j] = np.min(binary[(i - 1):(i + 2), (j - 1):(j + 2)])
            elif i == 0 and j == 0:
                erode[i, j] = np.min(binary[i:(i + 2), j:(j + 2)])
            elif i == 0:
                erode[i, j] = np.min(binary[i:(i + 2), (j - 1):(j + 2)])
            elif j == 0:
                erode[i, j] = np.min(binary[(i - 1):(i + 2), j:(j + 2)])

    cv2.imshow("my_erosion", erode)
    return erode


def dilation(image, binary_image):
    binary = binary_image
    dilate = np.zeros(image.shape)

    for i in range(0, 400):
        for j in range(0, 400):
            if i != 0 and j != 0:
                dilate[i, j] = np.max(binary[(i - 1):(i + 2), (j - 1):(j + 2)])
            elif i == 0 and j == 0:
                dilate[i, j] = np.max(binary[i:(i + 2), j:(j + 2)])
            elif i == 0:
                dilate[i, j] = np.max(binary[i:(i + 2), (j - 1):(j + 2)])
            elif j == 0:
                dilate[i, j] = np.max(binary[(i - 1):(i + 2), j:(j + 2)])

    cv2.imshow("my_dilate", dilate)
    return dilate


def morf_gradient(dilation, erosion):
    dilate = dilation
    erode = erosion

    result = np.absolute(np.array(dilate) - np.array(erode))
    cv2.imshow("My_morph_gradient", result)
    return result