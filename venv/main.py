import cv2
import numpy as np
import Power_transformation

#Гамма
gamma = 1.8

#Чтения изображения
image = cv2.imread("C:/Users/Acer/Desktop/Recognize/11.jpg")
cv2.imshow("Image_1", image)

#Преобразование в полутоновое
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image_2", gray_image)
print(gray_image[12][44])


#Степенное преобразование(гамма)
result_image = Power_transformation.gamma_2(gray_image, gamma)
cv2.imshow("Gamma", result_image)


#Оператор Робертса
mask = []
superimpose_mask_on_image

cv2.waitKey(0)
cv2.destroyAllWindows()