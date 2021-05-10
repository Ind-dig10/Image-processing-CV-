import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import cos, pi, sqrt

def DCT(image):
    img1 = image.astype('float')
    img_dct = cv2.dct(img1)

    img_dct_log = np.log(abs(img_dct))

    recor_temp = img_dct[0:40, 0:40]
    recor_temp2 = np.zeros(image.shape)
    recor_temp2[0:40, 0:40] = recor_temp

    # Восстановление сжатого изображения
    result = cv2.idct(recor_temp2)

    plt.subplot(221)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.imshow(image)
    plt.title('Исходное')

    plt.subplot(222)
    plt.imshow(img_dct_log)
    plt.title('dct')

    plt.subplot(223)
    plt.imshow(result)
    plt.title('Сжатие')

    plt.show()


def dct_2d(image, numberCoefficients=0):
    nc = numberCoefficients  # passando para NC para melhorar a visibilidade da formula
    height = image.shape[0]
    width = image.shape[1]
    imageRow = np.zeros_like(image).astype(float)
    imageCol = np.zeros_like(image).astype(float)

    for h in range(height):
        imageRow[h, :] = dct_1d(image[h, :], nc)  # aplicando IDCT na linhas
    for w in range(width):
        imageCol[:, w] = dct_1d(imageRow[:, w], nc)  # aplicando IDCT nas colunas

    return imageCol


def dct_1d(image, numberCoefficients=0):
    nc = numberCoefficients
    n = len(image)
    newImage = np.zeros_like(image).astype(float)

    for k in range(n):
        sum = 0
        for i in range(n):
            sum += image[i] * cos(2 * pi * k / (2.0 * n) * i + (k * pi) / (2.0 * n))
        ck = sqrt(0.5) if k == 0 else 1
        newImage[k] = sqrt(2.0 / n) * ck * sum

    # salvando os N maiores numeros e zerandos todos os outros
    if nc > 0:
        newImage.sort()
        for i in range(nc, n):
            newImage[i] = 0

    return newImage  # retorno de um VETOR


def idct_2d(image):
    height = image.shape[0]
    width = image.shape[1]
    imageRow = np.zeros_like(image).astype(float)
    imageCol = np.zeros_like(image).astype(float)

    for h in range(height):
        imageRow[h, :] = idct_1d(image[h, :])  # aplicando IDCT na linhas
    for w in range(width):
        imageCol[:, w] = idct_1d(imageRow[:, w])  # aplicando IDCT nas colunas

    return imageCol


def idct_1d(image):
    n = len(image)
    newImage = np.zeros_like(image).astype(float)

    for i in range(n):
        sum = 0
        for k in range(n):
            ck = sqrt(0.5) if k == 0 else 1  # operador tenario para verificar o valor do CK
            sum += ck * image[k] * cos(2 * pi * k / (2.0 * n) * i + (k * pi) / (2.0 * n))

        newImage[i] = sqrt(2.0 / n) * sum

    return newImage