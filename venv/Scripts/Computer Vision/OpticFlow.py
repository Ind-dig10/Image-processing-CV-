import numpy as np
import cv2 as cv


def getAngleDetector(maxCorners=45, qualityLevel=0.3, minDistance=7, blockSize=7):
    feature_params = dict(maxCorners=maxCorners,  # максимальное количество углов для возврата
                          qualityLevel=qualityLevel,  # минимально допустимое качество углов изображения
                          minDistance=minDistance,  # минимально возможное евклидово расстояние между углами
                          blockSize=blockSize)  # размер блока
    return feature_params


def getPointVector(angleDetector=None, grayFrame = None):
    points = []

    if angleDetector is None:
        points = np.zeros((10, 1, 2), dtype=np.float32)
        points[0] = [[340, 200]]
    else:
        p_all = cv.goodFeaturesToTrack(grayFrame, mask=None, **angleDetector)
        points = np.zeros((len(p_all), 1, 2), dtype=np.float32)
        points[0] = p_all[1]
        points[1] = p_all[2]
        points[3] = p_all[4]

    return points


def executeOpticFlow(video, color, angleDetector=None):
    cap = cv.VideoCapture(video)

    lk_params = dict(winSize=(15, 15),  # размер усредняющего окна гауссовой фильтрации
                     maxLevel=2,  # количество построенных уровней пирамиды разложения
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10,
                               0.03))  # условие выхода из итеративного процесса определения сдвига

    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    points = getPointVector(angleDetector, old_gray)
    mask = np.zeros_like(old_frame)

    while (True):
        ret, frame = cap.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Рассчет оптического потока
        new_points, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, points, None, **lk_params)
        print(err)

        if new_points is not None:
            good_new_point = new_points[st == 1]
            good_old_point = points[st == 1]

        for i, (new, old) in enumerate(zip(good_new_point, good_old_point)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color.tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color.tolist(), -1)
        img = cv.add(frame, mask)

        cv.imshow('frame', img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

        old_gray = frame_gray.copy()
        points = good_new_point.reshape(-1, 1, 2)

    cv.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    angleDetector = getAngleDetector()
    color = np.array([255,123,10]);
    executeOpticFlow('VID1.mp4', color, angleDetector)
