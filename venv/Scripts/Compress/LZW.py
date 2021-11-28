class LZW(object):
    # конструктор
    def __init__(self):
        self.dictionary = {}

    # первый шаг сжатия - создание словаря из всех элементов.
    def make_dict(self, string):

        # определяем все уникальные символы
        unique_char = list(set(string))

        # сортируем уникальные символы в алфавитном порядке
        unique_char = sorted(unique_char)

        # содаем словарь из символов вида "символ : порядковый_номер"
        #  например, d{'a'} -> '0' , d{'b'} -> '1' etc
        d = {i: str(j) for (i, j) in zip(unique_char, range(len(unique_char)))}

        # возвращаем словарь
        return d

    # методо сжатия
    def encode(self, string):

        # получаем словарь
        dictionary = self.make_dict(string)

        # делаем копию словаря
        self.dictionary = dictionary.copy()

        # добавляем к строке символ конца строки _
        string += '_'

        # результат
        answer = ""

        # сичтываем первый элемент
        X = string[0]

        # пробегаем строчку, начиная со второго элемента
        for i, Y in enumerate(string[1:]):

            # если конец строки, то выдать код текущей подстроки
            if Y == '_':
                answer += dictionary[X]

            # иначе
            else:
                # добавить в подстроку текущий символ
                XY = X + Y

                # если эта подстрока в словаре
                if XY in dictionary.keys():
                    # то запомнить подстроку для следующей итерации
                    X = XY
                # иначе
                else:
                    # выдать код текущей подстроки в результирующую строку
                    answer += dictionary[X]
                    # записать подстроку в словарь
                    dictionary[XY] = str(len(dictionary))

                    # текущий символ становиться новой подстрокой
                    X = Y

        # вернуть ответ
        return answer

    # метод для рахжатия. По сути обратный процесс
    def decode(self, string):
        # восстанавливаем исходный словарь - без тех записей, добавленных при
        # сжатии
        dictionary = self.dictionary.copy()

        # реверсируем словарь, то есть ключ становятся значениями, а значения
        # ключами, так как теперь по кодам нужно получать символы
        dictionary = {value: key for (key, value) in dictionary.items()}

        # символ конца строки
        string += '_'

        # ответ
        answer = ''

        # читаем первый символ
        X = string[0]

        # пробегаем сжатую строку, начиная с второго элемента
        for i, Y in enumerate(string[1:]):

            # если конец строки, то просто выдать символ
            if Y == '_':
                answer += dictionary[X]
            else:
                # новая подстрока
                XY = X + Y
                # если подстрока в словаре
                if XY in dictionary.keys():
                    # то подстроку меняем на текущую подтроку
                    X = XY
                else:
                    # иначе подстроку выдаем как результат
                    answer += dictionary[X]
                    # добавляем новую подстроку в словарь
                    dictionary[str(len(dictionary))] = dictionary[X] + dictionary[Y][0]

                    # текущий символ становиться подстрокой
                    X = Y
        # выдаем ответ
        return answer


string = 'wabbawabba'
print(f'Кодируемая строчка: {string}')
decoder = LZW()
a = decoder.encode(string)
print(f'Результат сжатия: {a}')
b = decoder.decode(a)
print(f'Результат разжатия: {b}')

