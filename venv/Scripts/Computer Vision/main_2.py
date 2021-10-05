from pylab import *
from PIL import Image
import Point_Detector as harris

"""
Example of detecting Harris corner points (Figure 2-1 in the book).
"""

# Импортировать изображение
im = array(Image.open('SD.jpg').convert('L'))

# Определить угловую точку Харриса
harrisim = harris.compute_harris_response(im)

# Функция ответа Харриса
harrisim1 = 255 - harrisim

figure()
gray()

# Нарисуйте график ответа Харриса
subplot(141)
imshow(harrisim1)
print(harrisim1.shape)
axis('off')
axis('equal')

threshold = [0.01, 0.05, 0.1]
for i, thres in enumerate(threshold):
    filtered_coords = harris.get_harris_points(harrisim, 6, thres)
    subplot(1, 4, i + 2)
    imshow(im)
    print(im.shape)
    plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
    axis('off')

# Модуль Харриса PCV в PCV, использованный в оригинальной книге
# harris.plot_harris_points(im, filtered_coords)

# plot only 200 strongest
# harris.plot_harris_points(im, filtered_coords[:200])

show()