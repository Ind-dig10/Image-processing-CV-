import cv2
import numpy as np
import argparse


# from __future__ import print_function

def GetBilinearPixel(imArr, posX, posY):
    return imArr[posX][posY]


def Affine(image, x, y):
    img = image

    num_rows, num_cols = img.shape[:2]

    translation_matrix = np.float32([[1, 0, x], [0, 1, y]])

    img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))

    cv2.imshow('Translation(OpenCV)', img_translation)


def CustomAffine(img, tx, ty):
    H, W, C = img.shape
    print(H)
    print(W)
    print(C)
    tem = img.copy()
    img = np.zeros((H + 2, W + 2, C), dtype=np.float32)
    img[1:H + 1, 1:W + 1] = tem

    H_new = np.round(H).astype(np.int)
    W_new = np.round(W).astype(np.int)
    out = np.zeros((H_new + 1, W_new + 1, C), dtype=np.float32)
    print(H_new)
    print(W_new)

    x_new = np.tile(np.arange(W_new), (H_new, 1))
    y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)
    print(x_new)
    print(y_new)

    x = np.round(x_new).astype(np.int) - tx
    y = np.round(y_new).astype(np.int) - ty
    print(x)
    print(y)

    x = np.minimum(np.maximum(x, 0), W + 1).astype(np.int)
    y = np.minimum(np.maximum(y, 0), H + 1).astype(np.int)

    out[y_new, x_new] = img[y, x]

    out = out[:H_new, :W_new]
    out = out.astype(np.uint8)

    return out
