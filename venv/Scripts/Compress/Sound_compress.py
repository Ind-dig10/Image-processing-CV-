# -*- coding: utf-8 -*-
import time
import numpy as np
# библиотека для wav файлов
from scipy.io.wavfile import read, write
# для нормализации данных
from sklearn.preprocessing import MinMaxScaler

# путь к файлу
path = 'raw.wav'


# открываем файл
def open_file(path):
    Fs, data = read(path)

    print(f"Размерность данных {data.shape}")
    print(f"Частота равна {Fs}")

    return data


# нормализация данных - то есть масштабирует данные для нового диапазона
def normalize(data, how):
    scaler = MinMaxScaler(how)

    scaler.fit(data)

    data = scaler.transform(data)

    return data


# mu сжатие - используем формулу и записываем файл
def mu_compress(array):
    array = normalize(array, (-1, 1))

    result = 1 / 8 * np.sign(array) * np.log(1 + 255 * np.abs(array))

    result = (result / 2 + 0.5) * 255

    result = np.uint8(result)

    write("mu.wav", 8000, result)

    return result


# А сжатие - используем формулу и записываем файл
def A_compress(array):
    s = normalize(array, (0, 1))

    A = 87.6

    result = np.zeros(s.shape)

    s[s == 0] = 0.00001

    result = np.where(1 / A < s, (1 + np.log(A * s)) / (1 + np.log(A)), A * s / (1 + np.log(A)))

    result = np.int8(result * 255)

    write("A.wav", 8000, result)

    return result


# дельта модуляция - записываем разницу
def delta_modulation(array):
    array = normalize(array, (0, 255))

    array = np.uint8(array)

    # вычисление приращения 
    result = np.diff(array, axis=0)
    write("DM.wav", 8000, result)

    return result


# Адаптивная
def ADCPM(array):
    index_table = [
        -1, -1, -1, -1, 2, 4, 6, 8,
        -1, -1, -1, -1, 2, 4, 6, 8
    ]

    step_table = [
        7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
        19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
        50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
        130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
        337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
        876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
        2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
        5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
        15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
    ]
    # для каждого канала вычисляются шаги, которые зависят от разницы.
    # две таблицы используются для кодирования

    step_index = 0

    prev_sample = array[0, 0]

    channel_0 = []
    channel_0.append(0)

    for sample in array[1:, 0]:
        sample = np.clip(sample, -32768, 32767)
        step_index = np.clip(step_index, 0, 88)

        step = step_table[step_index]
        diff = sample - prev_sample
        nibble = diff // step

        nibble = np.clip(nibble, 0, 15)

        channel_0.append(nibble)

        step_index = step_index + index_table[nibble]

    step_index = 0

    prev_sample = array[0, 1]

    channel_1 = []
    channel_1.append(0)

    for sample in array[1:, 1]:
        sample = np.clip(sample, -32768, 32767)
        step_index = np.clip(step_index, 0, 88)

        step = step_table[step_index]
        diff = sample - prev_sample
        nibble = diff // step

        nibble = np.clip(nibble, 0, 15)

        channel_1.append(nibble)

        step_index = step_index + index_table[nibble]

    channel_0 = np.uint8(channel_0)
    channel_1 = np.uint8(channel_1)

    result = np.dstack((channel_0, channel_1))[0]

    write('ADCPM.wav', 8000, result)

    return result


data = open_file(path)

start_time = time.time()
mu = mu_compress(data)
print("Время выполнения нелинейного u-типа", (time.time() - start_time))

start_time = time.time()
A = A_compress(data)
print("Время выполнения нелинейного А-типа", (time.time() - start_time))

start_time = time.time()
DM = delta_modulation(data)
print("Время выполнения дельта модулияции", (time.time() - start_time))

start_time = time.time()
channel_0 = ADCPM(data)
print("Время выполнения адаптивной ДИКМ", (time.time() - start_time))