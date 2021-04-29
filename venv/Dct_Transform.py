import cv2
import numpy as np
import matplotlib.pyplot as plt


def DCT(image):
    img1 = image.astype('float')
    img_dct = cv2.dct(img1)

    u, v = 8, 8
    widht = image.shape(1)
    height = image.shape(2)

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
