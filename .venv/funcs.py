"""
Файл с вспомогательными функциями
"""

import math
import numpy as np


# функция сортировки контура
def custom_sort(contour):
    return -contour.shape[0]


# функция сортировки координат
def polar_sort(item):
    return item[0][0]


# вычисление полярных координат
def get_polar_coordinates(x0, y0, x, y, center_x, center_y) -> tuple:
    # первая координата - радиус
    dx = center_x - x
    dy = center_y - y
    r = math.sqrt(dx*dx + dy*dy)

    # вторая координата - угол
    dx0 = center_x - x0
    dy0 = center_y - y0
    r0 = math.sqrt(dx0*dx0 + dy0*dy0)
    scalar_multiply = dx0*dx + dy0*dy
    cos_angle = scalar_multiply / r / r0
    sgn = dx0*dy - dx*dy0   # определение направления вектора

    # проверка величины косинуса
    if cos_angle > 1:
        if cos_angle > 1.0001:
            raise Exception("Something went wrong.. =(")
        cos_angle = 1

    angle = math.acos(cos_angle)
    if sgn < 0:
        angle = 2*math.pi - angle

    return angle, r


# Евклидово расстояние между двумя элементами
def get_length(item1, item2) -> int:
    _, point1 = item1
    _, point2 = item2
    x1, y1 = point1
    x2, y2 = point2
    dx = x1 - x2
    dy = y1 - y2
    r = math.sqrt(dx*dx + dy*dy)
    return r


# мини-макс нормализация вектора
def get_normalized_vector(arr_list: list) -> list:
    arr = np.array(arr_list)
    return (arr - arr.min()) / (arr.max() - arr.min())

