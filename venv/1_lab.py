import cv2
import numpy as np

max = 255.0
new_gray_image = gray_image/max
im_power_law_transformation = cv2.pow(new_gray_image, 0.5)
# plt_src = np.log(np.abs(spectre))

gamma = 1.5
adjusted = Power_transformation.Stepen(gray_image, gamma=gamma)
#cv2.imshow("gammam image 1", adjusted)
cv2.imshow("kzn", im_power_law_transformation)
"""
fourier transformation
    source: source image
    ctr_src: spectrum after fourier transformation and centralization
    plt_src: spectrum after log transformation
"""

"""
inverse fourier transformation
    src: spectrums
    ifft_img: the image after inverse fourier transformation
"""
def Stepen(image, gamma = 1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
					  for i in np.arange(0, 256)]).astype("uint8")

	return cv2.LUT(image, table)
