import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def fourier_trans(src):
    #дискретное преобразование фурье
    fftImage = np.fft.fft2(src)

    #Спектр после преобразования
    result = np.fft.fftshift(fftImage)

    return result


def obratnoe_fourier_trans(src):
    #Обратное дискретное преобразование
    spectre = np.fft.ifftshift(src)
    ifftImage = np.fft.ifft2(spectre)

    return ifftImage


def butterworth_high_pass_filter(source, D0=5, order=2):

    #filter_radius = D0
    img = source

    height, weight = img.shape
    center_h = int(height / 2)
    center_w = int(weight / 2)

    butterworth_high_pass_filter = np.zeros_like(img)

    for i in range(height):
        for j in range(weight):
            distFromCenter = np.sqrt(np.power((i - center_h), 2) + np.power((j - center_w), 2))  #Расстояние от точки D(v,u) до центра
            if distFromCenter != 0:
                butterworth_high_pass_filter[i][j] = 1 / (1 + np.power(D0 / distFromCenter, 2 * order))

    filtered_img = np.multiply(img, butterworth_high_pass_filter)

    return filtered_img

def Lab_3(img):
    img_path = img
    src = np.array(Image.open(img_path).convert("L"))
    fft_src = fourier_trans(src)


    img_list = [src]
    radius_list = ['Исходное', 2, 5, 15, 35, 1000]
    for i in radius_list[1:]:
        img_list.append(obratnoe_fourier_trans(butterworth_high_pass_filter(fft_src, i, 50)))

    img_list_name = radius_list

    _, img_xy = plt.subplots(2, 3, figsize=(12, 12))

    for i in range(2):
        for j in range(3):
            img_xy[i][j].imshow(np.abs(img_list[i * 3 + j]), cmap="gray")
            img_xy[i][j].set_title(img_list_name[i * 3 + j], size=20)
            img_xy[i][j].axis("off")

    plt.show()