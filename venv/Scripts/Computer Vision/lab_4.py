import numpy as np
import cv2

cap = cv2.VideoCapture('VID1.mp4')
# обучение по всему видео идет
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)
# среднее
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
# модуль разности
ret = True
while (ret):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dframe = cv2.absdiff(frame, grayMedianFrame)
    th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
    # th, dframe = cv2.threshold(dframe, 1, 255, cv2.THRESH_BINARY)
    cv2.imshow('frame', dframe)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.waitKey(0)

cap.release()

cv2.destroyAllWindows()

