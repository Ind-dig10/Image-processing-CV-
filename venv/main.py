import cv2
import numpy as np
#from Power_transformation import superimpose_mask_on_image
#from Roberts_2 import roberts_func
#from Binary import *
#from Aff_6 import smeshenie
#from lab3 import *
#from Dct_Transform import *
from Discrete_Cosine_Transform import *

#Гамма
gamma = 1.8

#Чтения изображения
image = cv2.imread("2.jpg", 0)

cv2.imshow("Image_1", image)
#Преобразование в полутоновое
#gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Image_2", gray_image)
#print(gray_image[12][44])

#1_лабораторная работа
#Степенное преобразование(гамма)
#result_image = Power_transformation.gamma_2(gray_image, gamma)
#cv2.imshow("Gamma", result_image)

#2_лабораторная работа
#Оператор Робертса
#roberts_func(image)
#kernely = np.array([[0, -1], [1, 0]], dtype=int)
#superimpose_mask_on_image(kernely, image)

#3_Лабораторная работа
#Lab_3("kzn_2.jpg")

#4_лабораторная работа
DCT(image)
#DCT(image)

#5 бинаризация изображения
#binaryImage = binary_image_transformations(image)
#Morphological_transformations(binaryImage, 5)
#Morphological_transformations(binaryImage, 1)

#6_Лабораторная работа
#Смещение изображения
#smeshenie(image)

cv2.waitKey(0)
cv2.destroyAllWindows()