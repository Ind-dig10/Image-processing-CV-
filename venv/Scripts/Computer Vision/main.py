import cv2
import numpy as np
import Contour_Recognize as cr


if __name__ == '__main__':
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
