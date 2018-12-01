# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 01:19:35 2018

@author: User
"""

import numpy as np
import scipy.stats
import random
import pandas as pd


def csv_to_arrays(filename):
    '''Функция для преобразования .csv таблицы в массивы numpy.

    Входные данные:
        filename - string, путь к таблице
    Выходные данные:
        X - array_like, массив x координат
        Y - array_like, массив y координат
    '''
    data = pd.read_csv(filename)  # считываем таблицу в датафрейм
    a = data.columns.tolist()  # получаем заголовки столбцов
    X_with_str = np.array(data[a[0]][2:])  # преобразуем столбцы в массивы
    Y_with_str = np.array(data[a[1]][2:])  # первые 3 строки без чисел
    # заменяем запятые на точки для дальнейшего преобразования в float
    X_with_comma = [x.replace(',', '.') for x in X_with_str]
    Y_with_comma = [y.replace(',', '.') for y in Y_with_str]
    # преобразуем в float
    X = np.array([float(x) for x in X_with_comma])
    Y = np.array([float(y) for y in Y_with_comma])
    return X, Y


def detrending(X, Y, power):
    '''Функция для удаления тренда порядка power из последовательности.

    Входные данные:
        X - array_like, массив x координат
        Y - array_like, массив y координат
        power - int, степень полинома
    Выходные данные:
        Y-Yfunc - array_like, массив y координат, приведенный к тренду
    '''
    # получаем коэффициенты полинома
    p = np.polyfit(X, Y, power)
    # получаем наилучшее приближение функции полиномом
    Yfunc = np.polyval(p, X)
    return Y - Yfunc


def distribution_creating(X, bins=10):
    '''Функция для преобразования последовательности случайных чисел в
    выборочную функцию плотности вероятности (гистограмму).

    Входные данные:
        X - array_like, массив случайных величин
        bins - int, число столбцов в гистограмме
    Выходные данные:
        x - list, список границ столбцов гистограммы
        Y - list, список значений вероятности для каждого столбца
    '''
    bins = bins + 1
    # находим минимальный и максимальный элемент массива для задания столбцов
    # гистограммы
    down_boundary = min(X)
    upper_boundary = max(X)
    # создаем массив с границами столбцов гистограммы
    x = np.linspace(down_boundary, upper_boundary + 0., bins)
    size = len(X)
    Y = []
    # считаем вероятности попасть в каждый из столбцов
    for i in range(1, bins):
        y = len([y for y in X if (y < x[i] and y >= x[i - 1])]) / size
        Y.append(y)
    # добавляем единицу к последнему столбцу, т. к. точка, соответствующая
    # верхней границе, не была добавлена в цикле
    Y[-1] += 1 / size
    return x, Y


def tail(X, Y, bins):
    '''Функция для получения хвоста распределения.

    Эта функция определяет хвост функции плотности вероятности, "отсекая" его
    на уровне в одну сигму.
    Входные данные:
        X - array_like, массив x координат
        Y - array_like, массив y координат
        bins - int, число столбцов гистограммы
    Выходные данные:
        X_tail - array_like, координаты столбцов хвоста гистограммы
        Y_tail - array_like, вероятности выпадения столбцов хвоста гистограммы
    '''
    # вычисляем выборочное среднее
    M = np.mean(Y)
    # вычисляем выборочное среднеквадратичное отклонение
    sigma = np.sqrt(sum((Y - M) ** 2) / len(Y))
    # создаем выборочную функцию плотности вероятности
    X, Y = distribution_creating(Y, bins)
    # 'отсекаем' хвост
    X_tail = [x for x in X if x > sigma + M]
    index = list(X).index(X_tail[0])
    Y_tail = np.array(Y[index:]) / sum(Y[index:])
    return X_tail, Y_tail


def pareto_pdf(x, alpha, sigma):
    '''Функция плотности вероятности распределения Парето (теоретическая).

    Входные данные:
        x - array_like, массив границ столбцов гистограммы
        alpha - float, параметр распределения
        sigma - float, параметр распределения
    Выходные данные:
        t - list, список со значениями вероятностей для каждого столбца
    '''
    t = []
    for i in range(len(x)-1):
        t.append(sigma ** alpha * (x[i] ** (-alpha) - x[i+1] ** (-alpha)))
    return t


def pareto_pdf_random(alpha, sigma, size, seed, bins):
    '''Функция плотности вероятности распределения Парето.

    Генерируется выборка из size распределенных по Парето случайных величин,
    для них составляется выборочная функция плотности вероятности.
    Входные данные:
        alpha - float, параметр распределения
        sigma - float, параметр распределения
        size - int, размер выборки
        seed - int, начальное состояние генератора
        bins - число отрезков для генерации выборочной функции плотности
        вероятности
    Выходные данные:
        x - list, список границ столбцов гистограммы
        Y - list, список значений вероятности для каждого столбца
    '''
    # создаем генератор случайных чисел, распределенных по Парето
    # генерируем size случайных чисел, распределенных по Парето, с параметрами
    # alpha и sigma - для тестирования алгоритма
    gen = scipy.stats.pareto
    test = gen.rvs(alpha, size=size, random_state=seed) * sigma ** alpha
    # преобразуем случайные величины в гистограмму
    x, Y = distribution_creating(test, bins)
    return x, Y


def main():
    '''Предполагается, что заданные последовательности имеют функцию плотности
    вероятности с тяжелым хвостом. Для оценки этого предположения предлагается
    следующий алгоритм:
        1.Из последовательности удаляется тренд
        2.Составляется выборочная функция плотности вероятности, и для нее
        вычисляются выборочное среднее и выборочная дисперсия
        3.От полученной функции 'отсекается' хвост на уровне в одну сигму
        4.Для хвоста методом Монте-Карло определяется значение показателей
        alpha и sigma
        5.При помощи alpha вычисляется значение показателя Хёрста
    '''
    filename = '1_60.csv'  # файл с данными
    power = 2
    eps = 1e-9  # задаем желаемую точность
    delta = 1  # начальное значение точности, для входа в цикл
    bins = 20  # число столбцов в гистограмме
    alpha = 0
    sigma = 0  # alpha, sigma - искомые параметры
    alpha_boundary = [1e-5, 2]  # задаем границы поиска alpha и sigma
    sigma_boundary = [1e-5, 2]
    # считываем координаты из файла
    X, Y = csv_to_arrays(filename)
    # убираем тренд
    Y = detrending(X, Y, power)
    # отделяем хвост функции плотности распределения
    x, Y = tail(X, Y, bins)
    steps = 0  # количество шагов, для предотвращения зацикливания
    while (delta > eps) and (steps < 100000):
        # на каждом шаге получаем случайные alp и sig
        alp = random.uniform(alpha_boundary[0], alpha_boundary[1])
        sig = random.uniform(sigma_boundary[0], sigma_boundary[1])
        # заполняем пустой список t теоретическими значениями вероятности
        # для данного распределения
        t = pareto_pdf(x, alp, sig)
        # вычисляем delta
        delta0 = sum((t - np.array(Y)) ** 2)
        # сохраняем значения alp и sig, если точность повысилась
        if delta0 < delta:
            delta = delta0
            sigma = sig
            alpha = alp
        steps += 1  # увеличиваем число пройденных шагов на один
    # выводим на экран найденные значения alpha и sigma
    print('sigma* = ', sigma)
    print('alpha* = ', alpha)
    print('steps = ', steps)
    print('delta = ', delta)
    # выводим на экран значение показателя Хёрста
    print('H = ', (3 - alpha) / 2)
    return 0


if __name__ == '__main__':
    main()
