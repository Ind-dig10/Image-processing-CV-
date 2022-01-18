import numpy as np
import cv2


def code_word(video_path, s):
    cap = cv2.VideoCapture(video_path)
    background_model = cv2.createBackgroundSubtractorMOG2() #Инициализируется модель фона

    while(1):
        ret, frame = cap.read()
        if ret == True:
            fgmask = background_model.apply(frame, s) # вычитание фона (вычисляет маску переднего плана)
            cv2.imshow('frame',fgmask) # fgmask - Выходная маска переднего плана в виде 8-битного двоичного изображения

            if (cv2.waitKey(30) & 0xff) == 27:
                break
    cap.release()
    cv2.destroyAllWindows()


code_word("VID1.mp4", 0.05)