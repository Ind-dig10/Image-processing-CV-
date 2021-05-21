import cv2
import numpy as np

img = cv2.imread('2.jpg', 0)
binary = np.zeros(img.shape)
threshold = 100
binary[img > threshold] = 255
binary[img <= threshold] = 0
cv2.imshow('binary_image', binary)
kernel = np.ones((3, 3))
erode = np.zeros(img.shape)
dilate = np.zeros(img.shape)
erode_cv = np.zeros(img.shape)
dilate_cv = np.zeros(img.shape)
print(np.min(binary[0:4, 0:4]))

for i in range(0, 400):
    for j in range(0, 400):
        if i != 0 and j != 0:
            erode[i, j] = np.min(binary[(i - 1):(i + 2), (j - 1):(j + 2)])
        elif i == 0 and j == 0:
            erode[i, j] = np.min(binary[i:(i + 2), j:(j + 2)])
        elif i == 0:
            erode[i, j] = np.min(binary[i:(i + 2), (j - 1):(j + 2)])
        elif j == 0:
            erode[i, j] = np.min(binary[(i - 1):(i + 2), j:(j + 2)])

for i in range(0, 400):
    for j in range(0, 400):
        if i != 0 and j != 0:
            dilate[i, j] = np.max(binary[(i - 1):(i + 2), (j - 1):(j + 2)])
        elif i == 0 and j == 0:
            dilate[i, j] = np.max(binary[i:(i + 2), j:(j + 2)])
        elif i == 0:
            dilate[i, j] = np.max(binary[i:(i + 2), (j - 1):(j + 2)])
        elif j == 0:
            dilate[i, j] = np.max(binary[(i - 1):(i + 2), j:(j + 2)])

cv2.imshow('dilate_image_andrey', dilate)
cv2.imshow('erode_image_andrey', erode)
print(binary[(0 - 1):(0 + 2)])

erode_cv = cv2.erode(binary, kernel)
dilate_cv = cv2.dilate(binary, kernel)
grad = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

cv2.imshow('erode_image_by_opencv', erode_cv)
cv2.imshow('dilate_image_by_opencv', dilate_cv)
cv2.imshow('GRAAAADIENT', grad)

result = np.absolute(np.array(dilate) - np.array(erode))
cv2.imshow("ff", result)

# cv2.imshow('diff_mask', abs(erode_cv-erode))
print("Diff between my erode and openCV erode: ", np.sum(erode != erode_cv))
while (True):
    if cv2.waitKey(33) == ord('a'):
        break
cv2.destroyAllWindows()
