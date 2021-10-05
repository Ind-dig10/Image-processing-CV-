from pylab import *
from numpy import *
from scipy.ndimage import filters


def compute_harris_response(im, sigma=3):
    "" "В изображении в оттенках серого рассчитайте функцию отклика детектора угла Харриса для каждого пикселя" ""
    # Вычислить производную
    imx = zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)
    # Вычислить компоненты матрицы Харриса
    Wxx = filters.gaussian_filter(imx * imx, sigma)
    Wxy = filters.gaussian_filter(imx * imy, sigma)
    Wyy = filters.gaussian_filter(imy * imy, sigma)
    # Вычислить собственные значения и следы
    Wdet = Wxx * Wyy - Wxy ** 2
    Wtr = Wxx + Wyy
    return Wdet / Wtr


def get_harris_points(harrisim, min_dist=10, threshold=0.1):
    "" "Возвращает угловые точки из изображения ответа Харриса. Min_dist - минимальное количество пикселей, которое разделяет угловые точки и границу изображения" ""
    # Найдите возможные угловые точки выше порога
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    # Получить координаты точки-кандидата
    coords = array(harrisim_t.nonzero()).T
    # И их значение ответа Харриса
    candidate_values = [harrisim[c[0], c[1]] for c in coords]
    # Сортировка баллов-кандидатов по значению ответа Харриса
    index = argsort(candidate_values)
    # Сохраняем позицию возможной точки в массив
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1
    # По принципу min_distance выбрать лучшую точку Харриса
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
    allowed_locations[(coords[i, 0] - min_dist):(coords[i, 0] + min_dist),

    (coords[i, 1] - min_dist):(coords[i, 1] + min_dist)] = 0
    return filtered_coords


def plot_harris_points(image, filtered_coords):
    "" "Нарисуйте обнаруженные угловые точки на изображении" ""
    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
    axis('off')
    show()
