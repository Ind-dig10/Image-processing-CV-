import cv2
import cv2 as cv
import matplotlib.pyplot as plt


image = cv.imread("kzn.jpg")
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

_, binary = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
image = cv.drawContours(image, contours, -1, (0, 255, 0), 2)
plt.imshow(image)
plt.show()