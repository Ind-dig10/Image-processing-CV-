import cv2
import numpy as np
import argparse

# from __future__ import print_function



def GetBilinearPixel(imArr, posX, posY):
    return imArr[posX][posY]


def MyFunc(image):
    img = image
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            pos = np.array([[i], [j], [1]], np.float32)
            # print pos
            pos = np.matmul(B, pos)
            r = int(pos[0][0])
            c = int(pos[1][0])
            # print r,c
            if (c <= 1024 and r <= 768 and c >= 0 and r >= 0):
                img_size[r][c] = img_size[r][c] + 1
                img_org[r][c] += GetBilinearPixel(img, i, j)

    for i in range(0, img_org.shape[0]):
        for j in range(0, img_org.shape[1]):
            if (img_size[i][j] > 0):
                img_org[i][j] = img_org[i][j] / img_size[i][j]

    cv2.imshow('Перевод', img_org)


def Smeshenie_2(imagee):
    # Получаю высоту и ширину
    image = cv2.cvtColor(imagee, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    dst_y, dst_x = np.indices((h, w))
    dst_lin_homg_pts = np.stack((dst_x.ravel(), dst_y.ravel(), np.ones(dst_y.size)))

    src_pts = np.float32([[693, 349], [605, 331], [445, 59]])
    dst_pts = np.float32([[1200, 1000], [1100, 1000], [1000, 500]])

    transf = cv2.getAffineTransform(dst_pts, src_pts)
    src_lin_pts = np.round(transf.dot(dst_lin_homg_pts)).astype(int)

    min_x, min_y = np.amin(src_lin_pts, axis=1)
    src_lin_pts -= np.array([[min_x], [min_y]])
    trans_max_x, trans_max_y = np.amax(src_lin_pts, axis=1)
    src = np.ones((trans_max_y + 1, trans_max_x + 1), dtype=np.uint8) * 127
    src[src_lin_pts[1], src_lin_pts[0]] = image.ravel()
    cv2.imshow('src', src)


def smeshenie(image):
    img = image
    l1 = 0
    l2 = 0
    num_rows, num_cols = img.shape[:2]
    M_matrix = np.float32([[1, 0, l1],
                           [0, 1, l2]])

    img_translation = cv2.warpAffine(img, M_matrix, (num_cols + 70, num_rows + 110))

    translation_matrix = np.float32([[1, 0, 150],
                                     [0, 1, 180]])

    img_translation = cv2.warpAffine(img_translation, translation_matrix, (num_cols + 400 + 30, num_rows + 110 + 50))

    cv2.imshow('Trans', img_translation)

def Affine(image, x, y):
    img = image

    num_rows, num_cols = img.shape[:2]

    translation_matrix = np.float32([[1, 0, x], [0, 1, y]])

    img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))

    cv2.imshow('Translation', img_translation)

