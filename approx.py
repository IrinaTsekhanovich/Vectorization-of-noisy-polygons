import numpy as np
import math as m
import cv2


def distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return m.sqrt(dx ** 2 + dy ** 2)


def calculate_significance(p1, p2, p3):
    b = distance(p1, p2)
    c = distance(p2, p3)
    a = distance(p1, p3)
    return b + c - a


def iterate_deleting(cnv, isEpsilon, value):
    n = len(cnv)
    result = []
    for i in range(n):
        result.append([cnv[i][0][0], cnv[i][0][1]])
    significance = [calculate_significance(result[n - 1], result[0], result[1])]
    for i in range(1, n - 1):
        significance.append(calculate_significance(result[i - 1], result[i], result[i + 1]))
    significance.append(calculate_significance(result[n - 2], result[n - 1], result[0]))
    m = 0
    if (isEpsilon):
        eps = value
        k = 2
    else:
        k = value
        eps = 1000
    while n > k and m < eps:
        argm = np.argmin(significance)
        m = min(significance)
        significance.pop(argm)
        result.pop(argm)
        n -= 1
        if argm == 0 or argm == n:
            significance[0] = calculate_significance(result[n - 1], result[0], result[1])
            significance[n - 1] = calculate_significance(result[n - 2], result[n - 1], result[0])
        elif argm - 1 == 0:
            significance[0] = calculate_significance(result[n - 1], result[0], result[1])
            significance[1] = calculate_significance(result[0], result[1], result[2])
        elif argm == n - 1:
            significance[n - 2] = calculate_significance(result[n - 3], result[n - 2], result[n - 1])
            significance[n - 1] = calculate_significance(result[n - 2], result[n - 1], result[0])
        else:
            significance[argm - 1] = calculate_significance(result[argm - 2], result[argm - 1], result[argm])
            significance[argm] = calculate_significance(result[argm - 1], result[argm], result[argm + 1])
    return result


def find_chaincode(x1, x2):
    if x2[0] == x1[0] and x2[1] == x1[1] + 1:
        return 0
    if x2[0] == x1[0] - 1 and x2[1] == x1[1] + 1:
        return 1
    if x2[0] == x1[0] - 1 and x2[1] == x1[1]:
        return 2
    if x2[0] == x1[0] - 1 and x2[1] == x1[1] - 1:
        return 3
    if x2[0] == x1[0] and x2[1] == x1[1] - 1:
        return 4
    if x2[0] == x1[0] + 1 and x2[1] == x1[1] - 1:
        return 5
    if x2[0] == x1[0] + 1 and x2[1] == x1[1]:
        return 6
    if x2[0] == x1[0] + 1 and x2[1] == x1[1] + 1:
        return 7


def calculate_aev(p1, p2, point):
    xi = p1[0]
    xj = p2[0]
    xk = point[0]
    yi = p1[1]
    yj = p2[1]
    yk = point[1]
    return ((xk - xi) * (yj - yi) - (yk - yi) * (xj - xi)) ** 2 / ((xi - xj) ** 2 + (yi - yj) ** 2)


def masood(cnv, isEpsilon, param):
    n = len(cnv)
    breakpoints = []
    for i in range(1, n - 2):
        if find_chaincode(cnv[i][0], cnv[i + 1][0]) != find_chaincode(cnv[i + 1][0], cnv[i + 2][0]) or \
                find_chaincode(cnv[i][0], cnv[i + 1][0]) != find_chaincode(cnv[i - 1][0], cnv[i][0]):
            breakpoints.append([cnv[i][0][0], cnv[i][0][1]])
    if find_chaincode(cnv[n - 2][0], cnv[n - 1][0]) != find_chaincode(cnv[n - 1][0], cnv[0][0]) or \
            find_chaincode(cnv[n - 2][0], cnv[n - 1][0]) != find_chaincode(cnv[n - 3][0], cnv[n-2][0]):
        breakpoints.append([cnv[n - 2][0][0], cnv[n - 2][0][1]])
    if find_chaincode(cnv[n - 1][0], cnv[0][0]) != find_chaincode(cnv[0][0], cnv[1][0]) or \
            find_chaincode(cnv[n - 1][0], cnv[0][0]) != find_chaincode(cnv[n-2][0], cnv[n-1][0]):
        breakpoints.append([cnv[n - 1][0][0], cnv[n - 1][0][1]])
    if find_chaincode(cnv[0][0], cnv[1][0]) != find_chaincode(cnv[1][0], cnv[2][0]) or \
            find_chaincode(cnv[0][0], cnv[1][0]) != find_chaincode(cnv[n-1][0], cnv[0][0]):
        breakpoints.append([cnv[0][0][0], cnv[0][0][1]])
    minerr = 0
    n = len(breakpoints)
    aev_list = [calculate_aev(breakpoints[n - 1], breakpoints[1], breakpoints[0])]
    for i in range(1, n - 1):
        aev_list.append(calculate_aev(breakpoints[i - 1], breakpoints[i + 1], breakpoints[i]))
    aev_list.append(calculate_aev(breakpoints[n - 2], breakpoints[0], breakpoints[n - 1]))
    if (isEpsilon):
        eps = param
        k = 2
    else:
        k = param
        eps = 1000
    while minerr < eps and n > k:
        argm = np.argmin(aev_list)
        minerr = min(aev_list)
        breakpoints.pop(argm)
        aev_list.pop(argm)
        n -= 1
        if argm == 0 or argm == n:
            aev_list[0] = calculate_aev(breakpoints[n - 1], breakpoints[1], breakpoints[0])
            aev_list[n - 1] = calculate_aev(breakpoints[n - 2], breakpoints[0], breakpoints[n - 1])
        elif argm - 1 == 0:
            aev_list[0] = calculate_aev(breakpoints[n - 1], breakpoints[1], breakpoints[0])
            aev_list[1] = calculate_aev(breakpoints[0], breakpoints[2], breakpoints[1])
        elif argm == n - 1:
            aev_list[n - 2] = calculate_aev(breakpoints[n - 3], breakpoints[n - 1], breakpoints[n - 2])
            aev_list[n - 1] = calculate_aev(breakpoints[n - 2], breakpoints[0], breakpoints[n - 1])
        else:
            aev_list[argm - 1] = calculate_aev(breakpoints[argm - 2], breakpoints[argm], breakpoints[argm - 1])
            aev_list[argm] = calculate_aev(breakpoints[argm - 1], breakpoints[argm + 1], breakpoints[argm])
    return breakpoints


def findY(line, x):
    return -(x[0]*line[0] + line[2])/line[1]


def cross(l1, l2):
    A1 = l1[0]
    B1 = l1[1]
    C1 = l1[2]
    A2 = l2[0]
    B2 = l2[1]
    C2 = l2[2]
    x = - (C1*B2 - C2*B1) / (A1*B2 - A2*B1)
    y = - (A1*C2 - A2*C1) / (A1*B2 - A2*B1)
    return [x, y]


def create_line(p1, p2):
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]
    A = y2 - y1
    B = x1 - x2
    C = (x2 - x1)*y1 - (y2 - y1)*x1
    return [A, B, C]


def find_angle(p1, p2):
    return m.atan((p2[1]-p1[1])/(p2[0]-p1[0]))


def approx(cnv, res_points, sizey):
    appr_cnv_first = cv2.approxPolyDP(cnv[0][1], 0.78, True)
    res_points.append([])
    for c in appr_cnv_first:
        res_points[len(res_points) - 1].append([c[0][0], sizey - c[0][1]])
    appr_masood_zero = masood(cnv[0][0], True, 0.6)
    res_points.append([])
    for c in appr_masood_zero:
        res_points[len(res_points) - 1].append([c[0], sizey - c[1]])
    appr_masood_2 = masood(cnv[0][2], True, 0.7)
    appr_masood_second = []
    for c in appr_masood_2:
        appr_masood_second.append([c[0], sizey - c[1]])

    # points 0,3 - big circle arc;1,2 - small circle arc
    outer_points = [[0, appr_masood_second[0][0], appr_masood_second[0][1]],
                    [0, appr_masood_second[0][0], appr_masood_second[0][1]],
                    [0, appr_masood_second[0][0], appr_masood_second[0][1]],
                    [0, appr_masood_second[0][0], appr_masood_second[0][1]]]
    for i in range(len(appr_masood_second)):
        if appr_masood_second[i][0] <= outer_points[0][1] and appr_masood_second[i][1] > outer_points[0][2]:
            outer_points[0] = [i, appr_masood_second[i][0], appr_masood_second[i][1]]
        if appr_masood_second[i][0] <= outer_points[1][1] and appr_masood_second[i][1] < outer_points[1][2]:
            outer_points[1] = [i, appr_masood_second[i][0], appr_masood_second[i][1]]
        if appr_masood_second[i][0] > outer_points[2][1] and appr_masood_second[i][1] <= outer_points[2][2]:
            outer_points[2] = [i, appr_masood_second[i][0], appr_masood_second[i][1]]
        if appr_masood_second[i][0] >= outer_points[3][1] and appr_masood_second[i][1] >= outer_points[3][2]:
            outer_points[3] = [i, appr_masood_second[i][0], appr_masood_second[i][1]]

    smth = []
    for i in range(outer_points[0][0] + 1, len(appr_masood_second)):
        smth.append(appr_masood_second[i])
    for i in range(outer_points[1][0]):
        smth.append(appr_masood_second[i])
    res_points.append(smth)
    smth = []
    for i in range(outer_points[2][0] + 1, outer_points[3][0]):
        smth.append(appr_masood_second[i])

    line0 = create_line([outer_points[0][1], outer_points[0][2]], [outer_points[1][1], outer_points[1][2]])
    line1 = create_line([outer_points[2][1], outer_points[2][2]], [outer_points[3][1], outer_points[3][2]])
    cntr = cross(line0, line1)  # centre of circle

    angle = find_angle([outer_points[2][1], outer_points[2][2]], [outer_points[3][1], outer_points[3][2]])
    outer_points[2][1] = (outer_points[1][2] - cntr[1]) * m.cos(angle) + cntr[0]
    outer_points[2][2] = (outer_points[1][2] - cntr[1]) * m.sin(angle) + cntr[1]

    line1 = create_line([outer_points[2][1], outer_points[2][2]], [outer_points[3][1], outer_points[3][2]])
    point1 = smth[0]
    point2 = smth[len(smth) - 1]

    while (smth[0][1] <= findY(line1, smth[0])):
        point1 = smth[0]
        smth.pop(0)
    while (smth[len(smth) - 1][1] <= findY(line1, smth[len(smth) - 1])):
        point2 = smth[len(smth) - 1]
        smth.pop(len(smth) - 1)
    line2 = create_line(smth[0], point1)
    line3 = create_line(smth[len(smth) - 1], point2)

    if point1 == smth[0]:
        line2 = create_line(smth[0], smth[1])
    else:
        line2 = create_line(smth[0], point1)
    if point2 == smth[len(smth) - 1]:
        line3 = create_line(smth[len(smth) - 1], smth[len(smth) - 2])
    else:
        line3 = create_line(smth[len(smth) - 1], point2)

    smth.insert(0, cross(line1, line2))
    smth.append(cross(line1, line3))
    res_points.append(smth)
    result = []
    for p in outer_points:
        result.append([p[1], p[2]])
    return result, cntr
