"""
Нахождение маркеров на изображении

FIX: на изображении есть блики (можно увидеть на бинарном изображении thresh_image),
которые считаются за контур и неправильно обрабатываются.

Я думаю, что надо подключить сюда алгоритм обнаружения близких точек, потому что
по бинарному изображению видно эти небольшие скосы, которые ну уж очень близко друг
к другу находятся.

В файле test_count_contours можно увидеть эти два выродка с бликами
(а еще интересно, что он тень считает за контур, но работает +- как надо)
"""

import math
import cv2
import numpy as np
import funcs

# инвариантный вектор
template_vector = np.array([0.02800683, 0., 0.66317664, 0.20533399, 0.01339446, 0.023891, 0.01481608, 0.90940451, 1.])

distance_thresh = 0.4

# читаем и обрабатываем изображение через медианный фильтр, перевод в ч/б
image = cv2.imread("images/many_markers.jpg")
filtered_image = cv2.medianBlur(image, 7)
image_gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

# порог бинаризации изображения
thresh = 100

ret, thresh_image = cv2.threshold(image_gray, thresh, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

image_contours = np.uint8(np.zeros((image.shape[0], image.shape[1])))

for selected_contour in contours:
    arclen = cv2.arcLength(selected_contour, True)

    # не рассматриваем малые объекты
    if arclen < 20:
        continue

    eps = 0.005
    epsilon = arclen * eps
    approx = cv2.approxPolyDP(selected_contour, epsilon, True)

    cv2.drawContours(image_contours, [approx], -1, (255, 255, 255), 1)

    if len(approx) == 9:
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
            r = math.sqrt(dx * dx + dy * dy)
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

        # создаем векторное описание
        size = len(polar_coordinates)
        lengths = []
        for i in range(size - 1):
            lengths.append(funcs.get_length(polar_coordinates[i], polar_coordinates[i + 1]))
        lengths.append(funcs.get_length(polar_coordinates[i - 1], polar_coordinates[0]))

        normalized_vector = funcs.get_normalized_vector(lengths)
        print(lengths)
        print(normalized_vector)

        # вычислим Евклидово расстояние
        square = np.square(normalized_vector - template_vector)
        sum_square = np.sum(square)
        distance = np.sqrt(sum_square)

        if distance < distance_thresh:
            for i in range(1, size):
                _, point1 = polar_coordinates[i - 1]
                _, point2 = polar_coordinates[i]
                x1, y1 = point1
                x2, y2 = point2
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=4)
            _, point1 = polar_coordinates[size - 1]
            _, point2 = polar_coordinates[0]
            x1, y1 = point1
            x2, y2 = point2
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=4)

cv2.imshow("Founded markers", image)
cv2.imshow("Image contours", image_contours)
cv2.imshow("Thresh image", thresh_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
