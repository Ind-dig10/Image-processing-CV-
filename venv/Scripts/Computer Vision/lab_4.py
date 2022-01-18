import numpy as np
import cv2


def code_word(video_path, s):
    cap = cv2.VideoCapture(video_path)
    background_model = cv2.createBackgroundSubtractorMOG2()

    while(1):
        ret, frame = cap.read()
        if ret == True:
            fgmask = background_model.apply(frame, s)
            cv2.imshow('frame', fgmask)

            if (cv2.waitKey(30) & 0xff) == 27:
                break
    cap.release()
    cv2.destroyAllWindows()


def background_averaging(video_path, frame):
    cap = cv2.VideoCapture(video_path)
    frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=frame)
    frames = []
    for id in frameIds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, id)
        ret, frame = cap.read()
        frames.append(frame)

    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

    ret = True
    while (ret):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dframe = cv2.absdiff(frame, grayMedianFrame)
        _, dframe = cv2.threshold(dframe, 25, 255, cv2.THRESH_BINARY)
        cv2.imshow('frame', dframe)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.waitKey(0)

    cap.release()

    cv2.destroyAllWindows()

code_word("VID1.mp4", 0.7)
#background_averaging("VID1.mp4", 10)
