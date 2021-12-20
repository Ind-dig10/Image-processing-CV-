# Детектор углов Ши-Томази и оптический поток Лукаса-Канаде

import numpy as np
import cv2 as cv

cap = cv.VideoCapture('VID1.mp4')

# параметры для определения угла Ши-Томазиp
feature_params = dict( maxCorners = 10, # максимальное количество углов для возврата
                       qualityLevel = 0.3, # минимально допустимое качество углов изображения (если лучший угол имеет показатель качества = 1500, а уровень качества = 0.01, то все углы с показателем качества меньше 15 игнорируются)
                       minDistance = 7, # минимально возможное евклидово расстояние между углами
                       blockSize = 7 ) # размер блока

# Параметры оптического потока Лукаса-Канад
lk_params = dict( winSize=(15, 15), # размер усредняющего окна гауссовой фильтрации
                  maxLevel = 2, # количество построенных уровней пирамиды разложения
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)) # условие выхода из итеративного процесса определения сдвига

# Выбор цветов для точек (маркеров)
color = np.random.randint(0, 255, (100, 3))

# Берёт первый кадр и находит в нем углы с помощью детектора углов Ши-Томази
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY) # 8-битное первое изображение
# вектор, содержащий координаты точек первого изображения, для которых должен быть рассчитан оптический поток
p_all = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Оставляем только одну точку в векторе
p0 = np.zeros((len (p_all),1,2),dtype=np.float32)
p0[1] = p_all[1]


# Инициализация маски для рисования
mask = np.zeros_like(old_frame)

while (1):
    ret, frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # 8-битное второе изображение

    # Рассчитавается оптический поток
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Отбирает хорошие точки
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    # Рисует траекторию движения точки
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)

    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # Обновление предыдущих кадров и предыдущих точк
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()
cap.release()