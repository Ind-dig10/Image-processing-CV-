import numpy as np
import cv2

cap = cv2.VideoCapture('VID1.mp4') # Загрузка видео
fgbg = cv2.createBackgroundSubtractorMOG2() # Инициализируется модель фона
k=0

while(1):
    ret, frame = cap.read()
    if ret == True:
        s = 0.005; # Скорость обучения
        fgmask = fgbg.apply(frame, s) # вычитание фона (вычисляет маску переднего плана)
        cv2.imshow('frame',fgmask) # fgmask - Выходная маска переднего плана в виде 8-битного двоичного изображения
        #cv2.resizeWindow('frame',600,600)
        cv2.imwrite(f'cb\\img_done_{k}.png', fgmask)
        k += 1

        if (cv2.waitKey(30) & 0xff) == 27:
            break
cap.release()
cv2.destroyAllWindows()