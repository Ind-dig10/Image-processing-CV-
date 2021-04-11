import cv2
import numpy as np


def gamma(gray_image, gamma=1.0):
    max = 255.0
    new_image = gray_image / max
    im_power_law_transformation = cv2.pow(new_image, gamma)

    return im_power_law_transformation


def gamma_2(gray_image, gamma=1.0):
    max = 255.0
    new_image = gray_image / max
    m, n = np.shape(gray_image)

    for i in range(m):
        for j in range(n):
            new_image[i][j] = pow(new_image[i][j], gamma)

    return new_image


def superimpose_mask_on_image(mask, image, color_delta=[20, -20, -20]):
    # fast, but can't handle overflows

    image[:, 0] = image[:, 0] + (mask[:, 0] / 255)
    image[:, 1] = image[:, 1] + (mask[:, 0] / 255)
    image[:, 2] = image[:, 2] + (mask[:, 0] / 255)


    cv2.imshow(" jcvkk",image)
