import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


def binary_image_transformations(im):
    threshold = 150
    imm = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    image = np.array(imm)

    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i, j] > threshold:
                image[i, j] = 255
            else:
                image[i, j] = 0

    cv2.imshow("Бинарное", image)
    return image


def Discrete_Transform(image):
    # img = image
    img = binary_image_transformations(image)
    height, width = image.shape[:2]
    u = 8
    v = u

    for i in range(height - 1):
        for j in range(width - 1):
            if image[i][j] == 0:
                cu = math.sqrt(1 / height)
                cv = cu
                img[i][j] = cu * cv * CosX(u, image[i][j], height) * CosX(v, image[i][j], height)
            else:
                cu = math.sqrt(2 / height)
                cv = cu
                CosX(u, image[i][j], height)
                img[i][j] = cu * cv * CosX(u, image[i][j], height) * CosX(v, image[i][j], height)

    plt.subplot()
    plt.imshow(img)
    plt.title('Сжатие')

    plt.show()
    # cv2.imshow(img)


def CosX(u, x, n):
    result = math.cos(((2 * x + 1) * u * math.pi) / 2 * n)
    return result


def CosY(v, y, n):
    result = math.cos(((2 * y + 1) * v * math.pi) / 2 * n)
    return result


def Dct(img):
    img1 = img.astype('float')

    C_temp = np.zeros(img.shape)
    dst = np.zeros(img.shape)

    m, n = img.shape
    N = n
    # C_temp[0, :] = 1 * np.sqrt(1 / N)

    for i in range(m):
        for j in range(n):
            if (i == 0):
                C_temp[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * N)) * np.sqrt(1 / N)
            else:
                C_temp[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * N)) * np.sqrt(2 / N)

    dst = np.dot(C_temp, img1)
    dst = np.dot(dst, np.transpose(C_temp))

    dct = np.log(abs(dst))

    img_recor = np.dot(np.transpose(C_temp), dst)
    img_recor1 = np.dot(img_recor, C_temp)

    img_dct = cv2.dct(img1)

    img_dct_log = np.log(abs(img_dct))

    img_recor2 = cv2.idct(img_dct)

    plt.subplot(231)
    plt.imshow(img1, 'gray')
    plt.title('Исходное изображение')
    plt.xticks([]), plt.yticks([])

    plt.subplot(232)
    plt.imshow(dct)
    plt.title('DCT1')
    plt.xticks([]), plt.yticks([])

    plt.subplot(233)
    plt.imshow(img_recor1, 'gray')
    plt.title('IDCT')
    plt.xticks([]), plt.yticks([])

    plt.subplot(234)
    plt.imshow(img, 'gray')
    plt.title('Исходное изображение')

    plt.subplot(235)
    plt.imshow(img_dct_log)
    plt.title('DCT2(OpenCV)')

    plt.subplot(236)
    plt.imshow(img_recor2, 'gray')
    plt.title('IDCT2(OpenCV)')

    plt.show()

