import cv2
import sys
import numpy as np
from scipy import ndimage
from PIL import Image

roberts_cross_v = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, -1]])

roberts_cross_h = np.array([[0, 0, 0],
                            [0, 0, 1],
                            [0, -1, 0]])


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    # note signed integer
    return np.asarray(img, dtype="int32")


def save_image(data, outfilename):
    img = Image.fromarray(np.asarray(np.clip(data, 0, 255), dtype="uint8"), "L")
    img.save(outfilename)


def roberts_cross(infilename, outfilename):
    image = load_image(infilename)

    vertical = ndimage.convolve(image, roberts_cross_v)
    horizontal = ndimage.convolve(image, roberts_cross_h)

    output_image = np.sqrt(np.square(horizontal) + np.square(vertical))

    save_image(output_image, outfilename)

    #for i in range(2):
    #    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    #    plt.title(titles[i])
    #    plt.xticks([]), plt.yticks([])
    #plt.show()

infilename = sys.argv[1]
outfilename = sys.argv[2]
roberts_cross(infilename, outfilename)
