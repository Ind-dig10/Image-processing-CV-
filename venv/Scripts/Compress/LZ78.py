from collections import namedtuple

Code = namedtuple('Code', ('position', 'next'))
string = 'abacababacabc'


class LZ78(object):

    # метод для кодирования, принимает на вход строчку
    def encode(self, string):

        # здесь используется словарь, который будет пополняться по ходу сжатия
        # Доступ к элементам словаря осуществляется через ключ, например
        # self.dictionary['a'] -> '3'
        # self это аналог this из других языков, то есть ссылка на объект класса
        self.dictionary = {}

        # добавляем пустую строку
        self.dictionary[''] = 0

        # обрабатываемая подстрока
        buffer = ''

        # список для хранения резуьтатов
        code_list = []

        # подстрока, используемая для запоминания подстроки с прошлой
        # итерации цикла
        old_buffer = ''

        # пробегаем элементы строки
        for i, s in enumerate(string):

            # добавляем текущий символ в подстроку
            buffer += s

            # альтернативный способ получения доступа к элементам словаря
            # возвращает элмент, если записи нет, то -1
            get = self.dictionary.get(buffer, -1)

            # если ключ не найден
            if get == -1:

                # если подстрока на прошлой итерации была в словаре
                if old_buffer:
                    # записать код прошлой подстроки и код текущего символа
                    code = Code(self.dictionary[old_buffer], string[i])

                    # сделать новую запись в словаре с текущей строкой
                    # в качестве метки используется длина словаря
                    self.dictionary[buffer] = len(self.dictionary)

                    # добавить код в список кодов
                    code_list.append(code)
                    # обнулить буффер
                    old_buffer = ''
                # если подстрока на прошлой итерации не была в словаре
                else:
                    # добавить подстроку в словарь
                    self.dictionary[buffer] = len(self.dictionary)

                    # записать код с 0 и текущим символом
                    code = Code(0, string[i])
                    # добавить код в список кодов
                    code_list.append(code)

                # обнулить буффер
                buffer = ''

            # если ключ найден
            else:
                # запоминаем текущую подстроку
                old_buffer = buffer
                old_get = get

        # вернуть список кодов
        return code_list

    # метод для декодирования, на входе список кодов
    def decode(self, code_list):

        # словарь с кодирования не сохраняется, так как мы его можем
        # восстановить сами

        self.dictionary = ['']

        # результат
        string = ""

        # пробегаем коды в списке кодов
        for i in code_list:
            # вычисляем подстроку
            word = self.dictionary[i.position] + i.next
            # добавляем подстроку в результат
            string += word

            # добавляем подстроку в словарь
            self.dictionary.append(word)

        return string


decoder = LZ78()
print(f'Кодируемая строчка: {string}')
encoded = decoder.encode(string)
print('Результат кодирования:')
for i in encoded:
    print(f'({i.position}, {i.next})')
decoded = decoder.decode(encoded)
print(f'Результат декодирования: {decoded}')
