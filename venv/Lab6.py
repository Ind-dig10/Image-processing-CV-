import cv2
import numpy as np
import argparse

# from __future__ import print_function

img_org = np.zeros(shape=(780, 1050))
img_size = np.zeros(shape=(780, 1050))

pts1 = np.float32([[693, 349], [605, 331], [445, 59]])
pts2 = np.float32([[1379, 895], [1213, 970], [684, 428]])
Mat = cv2.getAffineTransform(pts2, pts1)
B = Mat



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

# def smeshenie(image):
# parser = argparse.ArgumentParser(description='Code for Affine Transformations tutorial.')
# parser.add_argument('--input', help='Path to input image.', default='lena.jpg')
# args = parser.parse_args()
#    src = image #cv.imread(cv.samples.findFile(args.input))

#   srcTri = np.array([[0, 0], [src.shape[1] - 1, 0], [0, src.shape[0] - 1]]).astype(np.float32)
#  dstTri = np.array([[0, src.shape[1] * 0.33], [src.shape[1] * 0.85, src.shape[0] * 0.25],
#                     [src.shape[1] * 0.15, src.shape[0] * 0.7]]).astype(np.float32)
#v
# warp_mat = cv.getAffineTransform(srcTri, dstTri)
# warp_dst = cv.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))
# Rotating the image after Warp
# center = (warp_dst.shape[1] // 2, warp_dst.shape[0] // 2)
# angle = -90
# scale = 0.6
# rot_mat = cv.getRotationMatrix2D(center, angle, scale)
# warp_rotate_dst = cv.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))
# cv.imshow('Source image', src)
# cv.imshow('Warp', warp_dst)
# cv.imshow('Warp + Rotate', warp_rotate_dst)

def My_func3(image):
