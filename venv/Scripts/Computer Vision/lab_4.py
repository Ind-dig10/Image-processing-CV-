import cv2
import numpy as np
import os, sys

video = cv2.VideoCapture(os.path.join(ROOT_DIR, 'car_video.mp4'))
k = 0
imgs = []
while(True):
    ret, frame = video.read()
    if ret == False:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    frame = cv2.medianBlur(frame, 15)
    imgs.append(frame)
    if k == 0:
      print('enter')
      cv2_imshow(frame)
    k +=1
#or i in
img_mean = np.mean(imgs, axis = 0)
img_std = np.std(imgs, axis = 0)

print(img_mean)
print(img_std)

k = 0
for i in imgs:
    if np.any(i - img_mean > img_std):
        diff_image = cv2.absdiff(i, img_mean.astype("uint8"))
        #diff_image = np.mean(diff_image, axis=2)
        diff_image[diff_image<=img_mean] = 0
        diff_image[diff_image>img_mean] = 255
    if k < 5:
      cv2.imwrite(os.path.join(RESULT_DIR, 'frame' + str(k) + '.png'), diff_image)
      k +=1

print('success')
