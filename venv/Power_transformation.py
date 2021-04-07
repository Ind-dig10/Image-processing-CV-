import cv2
import numpy as np

def gamma(gray_image, gamma = 1.0):
	max = 255.0
	new_image = gray_image/max
	im_power_law_transformation = cv2.pow(new_image, gamma)

	return im_power_law_transformation


def gamma_2(gray_image, gamma = 1.0):
	max = 255.0
	new_image = gray_image/max
	m, n = np.shape(gray_image)

	for i in range(m):
		for j in range(n):
			new_image[i][j] = pow(new_image[i][j], gamma)

	return new_image

def superimpose_mask_on_image(mask, image, color_delta = [20, -20, -20], slow = False):
    # superimpose mask on image, the color change being controlled by color_delta
    # TODO: currently only works on 3-channel, 8 bit images and 1-channel, 8 bit masks

    # fast, but can't handle overflows
    if not slow:
        image[:,:,0] = image[:,:,0] + color_delta[0] * (mask[:,:,0] / 255)
        image[:,:,1] = image[:,:,1] + color_delta[1] * (mask[:,:,0] / 255)
        image[:,:,2] = image[:,:,2] + color_delta[2] * (mask[:,:,0] / 255)

    # slower, but no issues with overflows
    else:
        rows, cols = image.shape[:2]
        for row in xrange(rows):
            for col in xrange(cols):
                if mask[row, col, 0] > 0:
                    image[row, col, 0] = min(255, max(0, image[row, col, 0] + color_delta[0]))
                    image[row, col, 1] = min(255, max(0, image[row, col, 1] + color_delta[1]))
                    image[row, col, 2] = min(255, max(0, image[row, col, 2] + color_delta[2]))

    return