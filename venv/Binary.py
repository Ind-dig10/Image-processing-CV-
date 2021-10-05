import cv2
import math
import numpy as np
import time
from PIL import Image
from PIL import ImageEnhance


def threshold_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("input_gray_image", gray)

    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("threshold：%s" % ret)
    cv.imshow("OTSU", binary)


def Adaptive_Thresholding():
    img = cv2.imread('2.jpg', 0)
    _, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 2);
    return th2


def Adaptive_Thresholding_Custom(filename, step=20):
    img = Image.open(filename).convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.2)
    pixels = list(img.getdata())
    arr = np.array(pixels)
    arr2d = arr.reshape(img.size)

    blocks = np.reshape(arr2d, (-1, step, step))
    for block in blocks:
        factor = 1
        mean = np.mean(block)
        thresh = mean / factor

        block[block <= thresh] = 0
        block[block > thresh] = 1

    arr2d = np.reshape(blocks, (1, -1))

    img2 = Image.new("1", img.size)
    img2.putdata(arr2d[0].tolist())
    print(type(img2))
    return img2


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

    print(type(image))
    return image


def Morphological_transformations(image, type):
    kernel = np.ones((3, 3), np.uint8)

    result = {
        1: cv2.erode(image, kernel, iterations=1),  # Эрозия
        2: cv2.dilate(image, kernel, iterations=1),  # Наращивание
        3: cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel),
        4: cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel),
        5: cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel),  # Градиент
        6: cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel),
        7: cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    }
    default_value = -1
    res = result.get(type, default_value)
    return res


def erosion(image, binary_image):
    binary = np.array(binary_image)
    # binary = binary_image
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

    # cv2.imshow("my_erosion", erode)
    return erode


def dilation(image, binary_image):
    # binary = binary_image
    binary = np.array(binary_image)
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

    # cv2.imshow("my_dilate", dilate)
    return dilate


def morf_gradient(dilation, erosion):
    dilate = dilation
    erode = erosion

    result = np.absolute(np.array(dilate) - np.array(erode))
    cv2.imshow("My_morph_gradient", result)
    return result


def close_custom(image, binary):
    result = erosion(image, dilation(image, binary))
    cv2.imshow("Custom_CLOSE", result)


def testtt(image):
    h, w = image.shape
    newimg = np.array(imm)

    for i in range(h):
        for j in range(w):
            sum = 0
            avg = 0
            t = math.pow((3 * 2 + 1), 2)
            for n in range(i-3, i + 3):
                for m in range(j-3, j + 3):
                    if ( n < 0 or m < 0 or n > h - 3 or m > w - 3):
                        t -= 1
                        continue
                sum += newimg[n,m]
            avg = sum / t - 1
            if newimg[i, j] > avg:
                newimg[i, j] = 255
            else:
                newimg[i, j] = 0

    return newimg