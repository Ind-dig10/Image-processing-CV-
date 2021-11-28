# -*- coding: utf-8 -*-
from collections import namedtuple

# именнованный список - аналог струтуры. Здесь храниться код типа (0,0,b)
# но к полям мощно обращатьяс через имена - смещение, длина и следующий символ
Code = namedtuple('Code', ('offset', 'length', 'next'))
string = 'abacdbapcad'
print(f'Декодируемая строка: {string}')


#
class LZ77(object):

    # конструктор класса
    def __init__(self):
        # список для хранения кодов
        self.code_list = []

    # функция для получения следующего кода
    def search(self, string, i):
        # левая часть строки относительно обрабатываемого символа
        prefix = string[:i]

        # наш код, в начале пустой.
        code = None

        # пробегаем символы в строке правее обрабатываемого символа.
        for j in range(i, len(string)):
            # подстрока, которая будет искаться в префиксе
            sub_string = string[i:j + 1]

            # ищем подстроку в префиксе. index содержит индекс найденной подстроки,
            #  если не найдено, то -1
            index = prefix.find(sub_string)

            # если строка найдена
            if index != -1:
                # записываем код

                # проверяем выход за границу строки
                if j + 1 == len(string):
                    # записываем код - смещение, длина и следующий символ
                    code = Code(i - index, len(sub_string), '_')
                else:
                    # записываем код - смещение, длина и следующий символ
                    code = Code(i - index, len(sub_string), string[j + 1])

        # если код был найден
        if code:
            self.code_list.append(code)

            return code.length
        else:
            # если в не было найден ни одной подстроки
            # записваем нули
            code = Code(0, 0, string[i])
            self.code_list.append(code)

            return 0

    def encode(self, string):
        # функция для сжатия
        # пробегаем все строки и применяет функцию search

        # если мы нашли длинную подстроку, то её нет смысла обрабатывать снова.
        #  поэтому вводим переменную для пропуска значений
        length_to_skip = 0

        # списко для хранения кода
        self.code_list = []

        # пробегаем символы в строке
        for i in range(len(string)):
            # если нужно пропустить символы. то пропускаем
            if length_to_skip:
                length_to_skip -= 1
                # continue - досрочный переход к следующей итерации цикла
                continue

            # ищем следующий код
            length_to_skip = self.search(string, i)

        # получщенный список кодов возвращаем
        return self.code_list

        # функция для декодирования

    def decode(self, code_list):
        # здесь храним результат
        string = ''

        # пробегаем список кодов
        for i in code_list:

            # если смещение ноль, то пишем символ из поля "следующий символ"
            if i.offset == 0:
                string += i.next
            else:
                # если смещение есть, то ищем подстроку по смещению и длине серии

                # if для обработки небольшого затупа питона ( срез [-5:0] и [-5:]
                # работают по разному, из-за чего небольшая ручная обработка)
                if -i.offset + i.length != 0:
                    # к строке добавляем строку, полученная смещением
                    # для получения используется отрицательные индексы массива
                    # которые определяют символы с другого конца массива, то -2 это
                    # второй с конца элемент
                    string += string[-i.offset:-i.offset + i.length]
                else:
                    string += string[-i.offset:]

                # если символ конца, то конец
                if i.next != '_':
                    string += i.next

        # возврат результата
        return string


# создаем объеект класса
coder = LZ77()

# кодируем строчку, coded_string - это список
coded_string = coder.encode(string)

# декодируем код, decoded_string - строка
decoded_string = coder.decode(coded_string)

# выводим результат
print(f"Кодируемая строчка: {string}")
print(f"Результат сжатия :")
for i in coded_string:
    print(f'\t {i.offset} {i.length} {i.next}')

print(f"Результат разжатия : {decoded_string}")
