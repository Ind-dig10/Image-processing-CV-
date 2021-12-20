import cv2
import numpy as np
from Roberts_2 import roberts_func
from matplotlib import pyplot as plt


def waterched_custom(image):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)

    #markers = markers + 900

    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    cv2.namedWindow('first', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('second', unknown)
    cv2.imshow('first', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def watershed(image):
    img = cv2.imread(image, 0)
    imgc = cv2.imread(image)
    mark = np.zeros(img.shape[:2], dtype=np.int32)
    mark[400, 350] = 10
    mark[230, 200] = 20
    mark[1, 1] = 90

    water = cv2.watershed(imgc, mark)

    water *= 20
    water = water.astype(np.uint8)
    cv2.circle(water, (350, 350), 5, (255, 255, 0))
    cv2.circle(water, (230, 200), 5, (255, 255, 0))
    cv2.circle(water, (1, 1), 5, (255, 255, 0))
    cv2.imshow('threshold', imgc)

    cv2.imshow('water', water)
    while (True):
        if cv2.waitKey(33) == ord('q'):
            break
    cv2.destroyAllWindows()


def graf(image, n):
    img = cv2.imread(image)
    height, width = img.shape[:2]

    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (5, 5, width - 5, height - 5)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, n, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img1 = img * mask[:, :, np.newaxis]

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    cv2.imshow('image', img1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def laplassian(filename):
    ddepth = cv2.CV_16S
    kernel_size = 3
    src = cv2.imread(filename, cv2.IMREAD_COLOR)
    src = cv2.GaussianBlur(src, (3, 3), 0)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(src_gray, ddepth, ksize=kernel_size)
    abs_dst = cv2.convertScaleAbs(dst)

    print('Blur image')
    cv2.imshow(src)
    print('Laplassian image')
    cv2.imshow(abs_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def kmeans(image, k):
    imgc = cv2.imread(image)
    img = imgc.reshape((-1, 3))
    img = np.float32(img)

    ret, label, center = cv2.kmeans(img, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 5,
                         cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    ret = center[label.flatten()]
    r2 = ret.reshape((imgc.shape))

    cv2.imshow('kmeans', r2)
    while (True):
        if cv2.waitKey(33) == ord('q'):
            break
    cv2.destroyAllWindows()


def global_threshold(filename):
  img = cv2.imread(filename, 0)

  #Глобальный
  ret, img_tsd = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
  print('Binary thershold')
  cv2.imshow("global", img_tsd)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def adaptive_threshold(filename, block_cize):
    img = cv2.imread(filename, 0)

    #Адаптивный
    img = cv2.medianBlur(img,5)
    img_adv_tsd = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,block_cize,2)
    print('Adaptive thershold')
    cv2.imshow("adaptive", img_adv_tsd)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test(image):
    img = cv2.imread(image)
    gray = cv2.imread(image, 0)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Нахождение разницы
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1

    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    cv2.imshow("Image", img)

    while (True):
        if cv2.waitKey(33) == ord('q'):
            break
    cv2.destroyAllWindows()


def canny(image):
    img = cv2.imread(image, 0)
    edges = cv2.Canny(img, 200, 300)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Исходное изображение'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Метод Кэнни'), plt.xticks([]), plt.yticks([])
    plt.show()


image = 'flow.jpeg'
gray = cv2.imread(image)
#roberts_func(gray)
#canny(image)
#graf(image, 1)
#kmeans(image, 6)
#global_threshold(image)
#adaptive_threshold(image, 7)
#watershed(image)
waterched_custom(image)
