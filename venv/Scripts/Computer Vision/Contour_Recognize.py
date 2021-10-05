import cv2
import cv2 as cv
import matplotlib.pyplot as plt


def find_template_countour(filename):
    img = cv2.imread(filename, 0)
    img_for_drawing = cv2.imread(filename)
    *t, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_for_drawing, contours, 1, (0, 255, 0), 3)
    cv2.imshow('template', img_for_drawing)

    while (True):
        if cv2.waitKey(33) == ord('q'):
            cv2.destroyWindow('template')
            break

    return contours[1]


def find_main_image_contours(filename):
    main_image = cv2.imread(filename, 0)
    copy_image = cv2.imread(filename)
    main_image = cv2.medianBlur(main_image, 3)
    ret, main_image = cv2.threshold(main_image, 50, 200, cv2.THRESH_BINARY)
    *t, contours, hierarchy = cv2.findContours(main_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return copy_image, contours


def searching(contours, template):
    min = 10
    index = -1

    for i in range(0, len(contours)):
        matching = cv2.matchShapes(contours[i], template, 1, 0.0)

        if matching <= min:
            print(matching, i)
            min = matching
            index = i

    return index
