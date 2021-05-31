import cv2
import numpy as np
import matplotlib.pyplot as plt
#from Power_transformation import superimpose_mask_on_image
from Roberts_2 import roberts_func
from Binary import *
from Aff_6 import *
from test import *
from lab3 import *
from Dct_Transform import *
from Discrete_Cosine_Transform import *
from Power_transformation import *

#Гамма
gamma = 1.8

#Чтения изображения
#image1 = cv2.imread("2.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("2.jpg")
#image = cv2.imread("2.jpg").astype(np.float32)


filter_size = 4
#temp = np.zeros(image1.shape, image1.dtype)
#Преобразование в полутоновое
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("f", image)
#cv2.imshow("Image_2", gray_image)
#print(gray_image[12][44])

#1_лабораторная работа
#Степенное преобразование(гамма)
#result_image = gamma_2(gray_image, gamma)
#cv2.imshow("Gamma", result_image)

#2_лабораторная работа
#Оператор Робертса
#roberts_func(image)
#kernely = np.array([[0, -1], [1, 0]], dtype=int)


#3_Лабораторная работа
#Lab_3("kzn_2.jpg")

#4_лабораторная работа
#Dct(gray_image)

#5 бинаризация изображения
#binaryImage = binary_image_transformations(image)
#opencv_er = Morphological_transformations(binaryImage, 1)
#opencv_dl = Morphological_transformations(binaryImage, 2)
#opencv_mf = Morphological_transformations(binaryImage, 5)
#cv2.imshow("OPENCV_EROSION", opencv_er)
#cv2.imshow("OPENCV_DILATION", opencv_dl)
#cv2.imshow("OPENCV_MRF_GRADIENT", opencv_mf)

#er = erosion(image, binaryImage)
#dl = dilation(image, binaryImage)
#mf = morf_gradient(dl,er)

#plt.subplot(231)
#plt.imshow(opencv_er, 'gray')
#plt.title('opencv_erode')
#plt.xticks([]), plt.yticks([])

#plt.subplot(232)
#plt.imshow(opencv_dl, 'gray')
#plt.title('opencv_dilate')
#plt.xticks([]), plt.yticks([])

#plt.subplot(233)
#plt.imshow(opencv_mf, 'gray')
#plt.title('opencv_mrf-gradient')
#plt.xticks([]), plt.yticks([])

#plt.subplot(234)
#plt.imshow(er, 'gray')
#plt.title('Эрозия')

#plt.subplot(235)
#plt.imshow(dl)
#plt.title('дилатация')

#plt.subplot(236)
#plt.imshow(mf, 'gray')
#plt.title('Морфологический градиент')

#plt.show()

#Morphological_transformations(binaryImage, 1)

#6_Лабораторная работа
#Смещение изображения
Affine(image, 100, 50)  #Функция смещения от OpenCV. 1 Значение по горизонтали, 2 по вертикали
out = CustomAffine(image, tx=100, ty=50)  #Собственная функция смещения. 1 Значение по горизонтали, 2 по вертикали
cv2.imshow("Custom_Translate", out)

cv2.waitKey(0)
cv2.destroyAllWindows()