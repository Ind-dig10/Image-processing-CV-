import cv2
import math
import numpy as np
import time
from PIL import Image
from PIL import ImageEnhance


def thresholdIntegral(inputMat, s, T=1):
    # outputMat=np.uint8(np.ones(inputMat.shape)*255)
    outputMat = np.zeros(inputMat.shape)
    nRows = inputMat.shape[0]
    nCols = inputMat.shape[1]
    S = int(max(nRows, nCols) / 8)

    s2 = int(S / 4)

    for i in range(nRows):
        y1 = i - s2
        y2 = i + s2

        if (y1 < 0):
            y1 = 0
        if (y2 >= nRows):
            y2 = nRows - 1

        for j in range(nCols):
            x1 = j - s2
            x2 = j + s2

            if (x1 < 0):
                x1 = 0
            if (x2 >= nCols):
                x2 = nCols - 1
            count = (x2 - x1) * (y2 - y1)

            sum = s[y2][x2] - s[y2][x1] - s[y1][x2] + s[y1][x1]

            if ((int)(inputMat[i][j] * count) < (int)(sum * (1.0 - T))):
                outputMat[i][j] = 255
                # print(i,j)
            # else:
            #     outputMat[j][i] = 0
    return outputMat


def thresholdIntegral1(inputMat, s, T=0.15):
    # outputMat=np.uint8(np.ones(inputMat.shape)*255)
    outputMat = np.zeros(inputMat.shape)
    nRows = inputMat.shape[0]
    nCols = inputMat.shape[1]
    S = int(max(nRows, nCols) / 8)

    s2 = int(S / 4)

    for i in range(nRows):
        y1 = i - s2
        y2 = i + s2

        if (y1 < 0):
            y1 = 0
        if (y2 >= nRows):
            y2 = nRows - 1

        for j in range(nCols):
            x1 = j - s2
            x2 = j + s2

            if (x1 < 0):
                x1 = 0
            if (x2 >= nCols):
                x2 = nCols - 1
            count = (x2 - x1) * (y2 - y1)

            sum = s[y2][x2] - s[y2][x1] - s[y1][x2] + s[y1][x1]

            if ((int)(inputMat[i][j] * count) < (int)(sum * (1.0 - T))):
                outputMat[i][j] = 255
                # print(i,j)
            # else:
            #     outputMat[j][i] = 0
    return outputMat


if __name__ == '__main__':
    ratio = 1
    image = cv2.imdecode(np.fromfile('2.jpg', dtype=np.uint8), 0)
    img = cv2.resize(image, (int(image.shape[1] / ratio), int(image.shape[0] / ratio)), cv2.INTER_NEAREST)

    retval, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    cv2.namedWindow('OTSU threshold', 0)
    cv2.imshow('OTSU threshold', otsu)
    cv2.imwrite('otsu_results.jpg', otsu)

    # thresh = cv2.adaptiveThreshold(img,  255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    # retval, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_OTSU)
    # retval, thresh = cv2.threshold(img, retval, 255, cv2.THRESH_OTSU)

    time_start = time.time()
    roii = cv2.integral(img)
    time_end = time.time()
    print('integral cost', time_end - time_start)

    # time_start = time.time()
    for j in range(1):
        thresh = thresholdIntegral1(img, roii)
    time_end = time.time()
    print('totally cost', time_end - time_start)
    cv2.namedWindow('fast inergral threshold', 0)
    cv2.imshow('fast inergral threshold', thresh)
    cv2.imwrite('results.jpg', np.uint8(thresh))

    time_start = time.time()
    roii = cv2.integral(img)
    time_end = time.time()
    print('integral cost', time_end - time_start)

    # time_start = time.time()
    for j in range(1):
        thresh = thresholdIntegral(img, roii)
    time_end = time.time()
    print('totally cost', time_end - time_start)
    cv2.namedWindow('integral threshold', 0)
    cv2.imshow('integral threshold', thresh)

    img = cv2.imread('2.jpg', 0)
    _, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 2);
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2);

    cv2.imshow("Image", img)
    cv2.imshow("THRESH_BINARY", th1)
    cv2.imshow("ADAPTIVE_THRESH_MEAN_C", th2)
    cv2.imshow("ADAPTIVE_THRESH_GAUSSIAN_C", th3)


    def threshold(filename, step=20):
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
        img2.show()

        return img2


    threshold("2.jpg", 4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
