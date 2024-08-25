"""
Вычисление инвариантного представления (нормализованного вектора)
для обнаружения маркера
"""

import math
import cv2
import numpy as np
import funcs

# читаем и обрабатываем изображение через медианный фильтр, перевод в ч/б
image = cv2.imread("images/marker.jpg")
filtered_image = cv2.medianBlur(image, 7)
image_gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

# порог бинаризации изображения
thresh = 100

# бинаризация изображения и нахождение по нему контуров
_, thresh_image = cv2.threshold(image_gray, thresh, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# сортируем контуры в порядке убывания
contours = list(contours)
contours.sort(key=funcs.custom_sort)

selected_contour = contours[0]

# находим длину дуги контура (в пикселях)
arclen = cv2.arcLength(selected_contour, True)

# делаем аппроксимацию
eps = 0.005
epsilon = arclen * eps
approx = cv2.approxPolyDP(selected_contour, epsilon, True)

# рисуем аппроксимированные точки на начальном изображении
canvas = image.copy()
for point in approx:
    cv2.circle(canvas, (point[0][0], point[0][1]), 7, (0, 255, 0), -1)

# находим центральную точку (среднее арифметическое всех координат)
sum_x = 0.0
sum_y = 0.0
for point in approx:
    sum_x += float(point[0][0])
    sum_y += float(point[0][1])
center_x = sum_x / float(len(approx))
center_y = sum_y / float(len(approx))

# находим точку, наиболее удаленную от центра
maximum = 0
beg_point = 1
for i in range(0, len(approx)):
    point = approx[i]
    x = float(point[0][0])
    y = float(point[0][1])
    dx = x - center_x
    dy = y - center_y
    r = math.sqrt(dx*dx + dy*dy)
    if r > maximum:
        maximum = r
        beg_point = i

# создаем массив полярных координат
polar_coordinates = []
x0 = approx[beg_point][0][0]
y0 = approx[beg_point][0][1]
for point in approx:
    x = int(point[0][0])
    y = int(point[0][1])
    angle, r = funcs.get_polar_coordinates(x0, y0, x, y, center_x, center_y)
    polar_coordinates.append(((angle, r), (x, y)))
polar_coordinates.sort(key=funcs.polar_sort)

# рисуем контуры так, чтобы было видно, как проходит отрисовка:
# каждый новый контур рисуется более толстым
image_contours = np.uint8(np.zeros((image.shape[0], image.shape[1])))
size = len(polar_coordinates)
for i in range(1, size):
    _, point1 = polar_coordinates[i-1]
    _, point2 = polar_coordinates[i]
    x1, y1 = point1
    x2, y2 = point2
    cv2.line(image_contours, (x1, y1), (x2, y2), 255, thickness=i)
_, point1 = polar_coordinates[size-1]
_, point2 = polar_coordinates[0]
x1, y1 = point1
x2, y2 = point2
cv2.line(image_contours, (x1, y1), (x2, y2), 255, thickness=size)

lengths = []
for i in range(size - 1):
    lengths.append(funcs.get_length(polar_coordinates[i], polar_coordinates[i+1]))
lengths.append(funcs.get_length(polar_coordinates[i-1], polar_coordinates[0]))
print(funcs.get_normalized_vector(lengths))

cv2.circle(image_contours, (int(center_x), int(center_y)), 15, (255, 255, 255), 3)

cv2.imshow("Original image", canvas)
cv2.imshow("Contours image", image_contours)

cv2.waitKey(0)
cv2.destroyAllWindows()
