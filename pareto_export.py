# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 01:19:35 2018

@author: User
"""

import numpy as np
import scipy.stats
import random


def distribution_creating(X, bins=10):
    '''Функция для преобразования последовательности случайных чисел в
    выборочную функцию распределения (гистограмму).

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
    down_border = min(X)
    upper_border = max(X)
    # создаем массив с границами столбцов гистограммы
    x = np.linspace(down_border, upper_border + 0., bins)
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


def main():
    eps = 1e-7  # задаем желаемую точность
    delta = 1  # начальное значение точности, для входа в цикл
    bins = 20  # число столбцов в гистограмме
    alpha0 = 1.8
    sigma0 = 1  # alpha0, sigma0 - показатели тестового распределения
    alpha = 0
    sigma = 0  # alpha, sigma - искомые параметры
    size = 1000  # размер тестовой выборки
    nums = 10  # число проходов с вычислением параметров
    alpha_bord = [1e-5, 2]  # задаем границы поиска alpha и sigma
    sigma_bord = [1e-5, 2]
    '''Создаем объект класса распределения
    pareto - распределение Парето первого типа, p(x) =
    alpha / x ^ (alpha + 1),
    rvs - функция для генерации распределенных согласно классу случайных
    величин, со следующими параметрами:
        c - float, alpha, параметр распределения
        loc - float, mu, параметр распределения
        scale - float, sigma, параметр распределения
        size - int, размерность возвращаемого списка
        random_state - int, начальное состояние генератора
    Возвращает:
        test - list, список случайных величин размера size
    '''
    Alp = []
    # генерируем size случайных чисел, распределенных по Парето, с параметрами
    # alpha0 и sigma0 - для тестирования алгоритма
    gen = scipy.stats.pareto
    test = gen.rvs(alpha0, size=size, random_state=200) * sigma0 ** alpha0
    # преобразуем случайные величины в гистограмму
    x, Y = distribution_creating(test, bins)
    # цикл для усреднения результата
    for k in range(nums):
        steps = 0  # количество шагов, для предотвращения зацикливания
        while (delta > eps) and (steps < 50000):
            # на каждом шаге получаем случайные alp и sig
            alp = random.uniform(alpha_bord[0], alpha_bord[1])
            sig = random.uniform(sigma_bord[0], sigma_bord[1])
            t = []
            # заполняем пустой список t теоретическими значениями вероятности
            # для данного распределения
            for i in range(len(x)-1):
                t.append(sig ** alp * (x[i] ** (-alp) - x[i+1] ** (-alp)))
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
        delta = 1
        Alp.append(alpha)
    print('<alpha> = ', np.mean(Alp))
    return 0


if __name__ == '__main__':
    main()
