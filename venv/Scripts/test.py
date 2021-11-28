# загрузим библиотеки
import numpy as np
import cv2
import pandas as pd
import time
import os, sys

ROOT_DIR = os.path.abspath("")


def load_table_keys(path='table.csv'):
    # открыть файл и считать таблицу
    table = pd.read_csv(os.path.join(ROOT_DIR, 'table.csv'), sep=',', encoding="utf-8",
                        dtype={'Белые коды': str, 'Черные коды': str})
    # из таблицы сделать два словаря
    white = dict(zip(table['Длина\n серии'], table['Белые коды']))
    black = dict(zip(table['Длина\n серии'], table['Черные коды']))

    # отдельно получаем список ключей - для удобства
    keys = list(white.keys())

    # упаковываем словари в список
    table = [white, black]

    # создаём обратную таблицу
    rev_white = {}
    rev_black = {}

    for i in white:
        j = white[i]

        rev_white[bytes(j, 'utf-8')] = i

    for i in black:
        j = black[i]

        rev_black[bytes(j, 'utf-8')] = i

    rev_table = [rev_white, rev_black]

    return table, rev_table, keys


def load_image(image_name='image.PNG'):
    image = cv2.imread(os.path.join(ROOT_DIR, image_name), 0)
    _, mono_image = cv2.threshold(image, 126, 1, cv2.THRESH_BINARY)
    return mono_image


def count_series(row):
    element = row[0]
    count = 1

    for i in row[1:]:
        if i == element:
            count += 1
        else:
            break
    return (element, count)


def coding(image):
    print('Начинаем сжатие')

    # записываем время начала - для вычисления затраченного времени
    started = time.time()

    # создаем бинарный файл для записи
    binary = open(os.path.join(ROOT_DIR, 'image.fax'), 'wb')

    # для каждой строки изображения
    for row in image:
        # вычисляем первую серию
        element, count = count_series(row)

        # код подразумевает периодичность черных и белых серий,, поэтому если
        # первый пел черный, то добавляется пустой белый пел в начале
        if element == 1:
            coded_row = b'000111'
        else:
            coded_row = b''

        # пока не обработали всю строку
        while (row.any()):
            # вычисляем серию
            element, count = count_series(row)
            # отбрасываем обработанную серию
            row = row[count:]

            # кодируем серию
            if count in keys:
                # если длина серии есть в таблице
                if count < 64:
                    # если меньше 64, то берём из таблицы
                    byte_chunk = bytes(table[element][count], 'utf-8')
                    coded_row += byte_chunk
                else:
                    # если табличное значение кратно 64, добавляем пустой пэл после
                    # так советовали делать в более полной статье
                    byte_chunk = bytes(table[element][count], 'utf-8')
                    coded_row += byte_chunk

                    byte_chunk = bytes(table[element][0], 'utf-8')
                    coded_row += byte_chunk

            else:
                # если длины серии нет в таблице, то вычисляем её как сумму двух серий
                # одна кратна 64, другая меньше 64. и записываем оба кода друг за другом
                for key in keys[::-1]:
                    if key <= count:
                        byte_chunk = bytes(table[element][key], 'utf-8')
                        coded_row += byte_chunk
                        count -= key
                        if count == 0:
                            break

        # код конца строки
        coded_row += b'000000000001'

        # записываем полученную строчку в файл
        binary.write(coded_row)

    # закрываем файл
    binary.close()

    print(f'Заканчиваем сжатие. Времени затрачено {time.time() - started} секунд')


def decode_pel(binary, next_pel_color):
    # читаем строчку
    bin_string = b''

    for i in binary:
        # питон автоматические переводит биты в инт, поэтому вручную меняю на
        # битовое представление.
        if i == 48:
            i = b'0'
        elif i == 49:
            i = b'1'

        bin_string += i

        # считываем слово, пока не считается код из таблицы

        # next_pel_color - определяет текущий цвет серии
        if next_pel_color == 0:
            # если код есть в таблице
            if bin_string in rev_table[0]:
                # запоминаем слово
                first_code = bin_string
                # если длина серии меньше 64 - можем не тратить время на второй код
                # и сразу вернуть найденную длину серии
                if rev_table[next_pel_color][first_code] < 64:
                    return 0, rev_table[0][first_code], binary[len(bin_string):], 1

                break

                # если конец файл - выйти
            elif bin_string == b'000000000001':
                return -1, 0, binary[len(bin_string):], -1

        # аналогично для черных серий
        elif next_pel_color == 1:
            if bin_string in rev_table[1]:
                first_code = bin_string
                if rev_table[next_pel_color][first_code] < 64:
                    return 1, rev_table[1][first_code], binary[len(bin_string):], 0
                break
            elif bin_string == b'000000000001':
                return -1, 0, binary[len(bin_string):], -1

        # случай если что-то пошло не так - длина кода не может быть больше 12
        elif len(bin_string) > 13:
            print("error, too long")
            return -1, -1, -1, -1

    # так как иногда записываются два кода ( когда больше 63 и длины нет в таблице),
    #  то ищем второе слово. Код аналогичен предыдущему.
    bin_string = b''

    for i in binary[len(first_code):]:
        if i == 48:
            i = b'0'
        elif i == 49:
            i = b'1'

        bin_string += i

        if next_pel_color == 0:
            if bin_string in rev_table[0]:
                second_code = bin_string

                return 0, rev_table[0][first_code] + rev_table[0][second_code], binary[
                                                                                len(first_code + second_code):], 1
            elif bin_string == b'000000000001':
                return 0, rev_table[0][first_code], binary, 1

        elif next_pel_color == 1:
            if bin_string in rev_table[1]:
                second_code = bin_string
                return 1, rev_table[1][first_code] + rev_table[1][second_code], binary[
                                                                                len(first_code + second_code):], 0
            elif bin_string == b'000000000001':
                return 1, rev_table[0][first_code], binary, 0

        elif len(bin_string) > 13:
            print("error, too long")
            return -1, -1, -1, -1


def decoding(file_name='image.fax'):
    print('Начинаем разжатие')
    started = time.time()

    # открываем бинарный файл для чтения
    binary = open(os.path.join(ROOT_DIR, file_name), 'rb')
    # считываем данные
    binary = binary.read()

    # небольшое удобство - список декодированных пикселей
    image_list = []

    # пока не прочитали весь бинарный код
    while binary:

        #  список декодированных пикселей строки
        L = []

        # считываем первый код - если пустой белый, то первая серия черная
        # иначе - белая.
        chunk = binary[:6]

        if chunk != b'000111':
            next_pel_color = 0
        else:
            binary = binary[6:]
            next_pel_color = 1
        while True:
            # декодириуем пел
            element, series, binary, next_pel_color = decode_pel(binary, next_pel_color)

            # если вся строка обработана
            if element == -1:
                # записываем наш ряд в список
                image_list.append(L)
                break
            elif element == 0:
                # в список L добавляем столько нулей, сколько
                # в серии
                L += [0] * series
            elif element == 1:
                # в список L добавляем столько единиц, сколько
                # в серии
                L += [1] * series

    print(f'Заканчиваем разжатие. Времени затрачено {time.time() - started} секунд')
    # преобразуем список в массив
    return np.asarray(image_list)


table, rev_table, keys = load_table_keys()
image = load_image()
coding(image)
my_image = decoding()
type(my_image)
our_img_size = os.path.getsize(os.path.join(ROOT_DIR, 'image.fax'))
img_size = os.path.getsize(os.path.join(ROOT_DIR, 'image.bmp'))

print(f'Исходный размер несжатого изображения: {img_size} байт')
print(f'Размер сжатого изображения: {our_img_size} байт')

print(f'Коэффициент сжатия: {img_size / our_img_size}')
