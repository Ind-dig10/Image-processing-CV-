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
    #img = image
    img = binary_image_transformations(image)
    height, width = image.shape[:2]
    u = 8
    v = u

    for i in range(height - 1):
        for j in range(width - 1):
            print(image[i][j])
            #if image[i][j] == 0:
            #    cu = math.sqrt(1 / height)
            #    cv = cu
            #    img[i][j] = cu * cv * CosX(u, image[i][j], height) * CosX(v, image[i][j], height)
            #else:
            #    cu = math.sqrt(2 / height)
            #    cv = cu
            #    CosX(u, image[i][j], height)
            #    img[i][j] = cu * cv * CosX(u, image[i][j], height) * CosX(v, image[i][j], height)

    plt.subplot()
    plt.imshow(img)
    plt.title('Сжатие')

    plt.show()
    #cv2.imshow(img)


def CosX(u, x, n):
    result = math.cos(((2 * x + 1) * u * math.pi) / 2 * n)
    return result


def CosY(v, y, n):
    result = math.cos(((2 * y + 1) * v * math.pi) / 2 * n)
    return result
