import cv2
import numpy as np
import Contour_Recognize as cr

image = None
video = None


def lab_1():
    image, image_contours = cr.find_main_image_contours('source/11.jpg')
    A_template = cr.find_template_countour('source/aaa.jpg')
    I_template = cr.find_template_countour('source/ttt.jpg')

    A_index = cr.searching(image_contours, A_template)
    cv2.drawContours(image, image_contours, A_index, (0, 255, 0), 3)

    I_index = cr.searching(image_contours, I_template)
    cv2.drawContours(image, image_contours, I_index, (0, 255, 0), 3)

    cv2.imshow('result', image)

    while (True):
        if cv2.waitKey(33) == ord('q'):
            break
    cv2.destroyAllWindows()


def load_image(image_path):
    return cv2.imread(image_path)


if __name__ == '__main__':
    img = load_image()
    cv2.imshow("result", img)
    cv2.waitKey()