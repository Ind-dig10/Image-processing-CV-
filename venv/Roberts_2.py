import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def roberts_func(inputImage):
    # Чтение изображения
    img = inputImage

    grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)

    x = cv.filter2D(grayImage, cv.CV_16S, kernelx)
    y = cv.filter2D(grayImage, cv.CV_16S, kernely)

    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)

    #Вычисление взвешенной суммы
    Roberts = cv.addWeighted(absX, 1, absY, 0.8, -100)

    print(absX)
    print(absY)
    # cv.imshow("Ishodnoe", rgb_img)
    cv.imshow("Roberts", Roberts)


