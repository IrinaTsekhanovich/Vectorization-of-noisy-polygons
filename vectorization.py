import numpy as np
import math as m
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import gmsh
from collections import deque
from PIL import Image, ImageDraw

def findcontours_suzuki(data, symbol):
    img = []
    for i in range(len(data)):
        img.append([])
        for j in range(len(data[0]) - 1):
            if data[i][j] == symbol:
                if not (not (j > 1 and j < len(data[0]) - 1 and data[i][j + 1] != symbol and data[i][
                    j - 1] != symbol) and not (
                        i > 1 and i < len(data) - 1 and data[i + 1][j] != symbol and data[i - 1][j] != symbol)):
                    img[i].append(0)
                else:
                    img[i].append(255)
            else:
                img[i].append(0)
    image = np.uint8(img)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours


def isCNV_old(data, i, j):
    if i == 0 or j == 0 or i == len(data) - 1 or j == len(data[0]) - 2 or (
            (j + 1 < len(data[0]) - 1 and data[i][j + 1] != data[i][j]) or
            (j + 1 < len(data[0]) - 1 and i - 1 >= 0 and data[i - 1][j + 1] != data[i][j]) or
            (j + 1 < len(data[0]) - 1 and i + 1 < len(data) and data[i + 1][j + 1] != data[i][j]) or
            (i + 1 < len(data) and data[i + 1][j] != data[i][j]) or
            (i + 1 < len(data) and j - 1 >= 0 and data[i + 1][j - 1] != data[i][j]) or
            (j - 1 >= 0 and data[i][j - 1] != data[i][j]) or
            (j - 1 >= 0 and i - 1 >= 0 and data[i - 1][j - 1] != data[i][j]) or
            (i - 1 >= 0 and data[i - 1][j] != data[i][j])):
        return True
    else:
        return False

def find_neighbour(S,i,j):
    if S == 0:
        return i, j+1
    if S == 1:
        return i-1, j+1
    if S == 2:
        return i-1, j
    if S == 3:
        return i-1, j-1
    if S == 4:
        return i, j-1
    if S == 5:
        return i+1, j-1
    if S == 6:
        return i+1, j
    if S == 7:
        return i+1, j+1


def findcontours_teo(start_point, img, symbol):
    first = True
    current_point = start_point
    S = 6
    cnv = [start_point]
    while (first or current_point != start_point):
        found = False
        while not found:
            if S == 0:
                i1, j1 = find_neighbour(7, current_point[0], current_point[1])
            else:
                i1, j1 = find_neighbour((S - 1) % 8, current_point[0], current_point[1])
            i2, j2 = find_neighbour(S, current_point[0], current_point[1])
            i3, j3 = find_neighbour((S + 1) % 8, current_point[0], current_point[1])
            if img[i1][j1] == symbol or img[i1][j1] >= 3:
                found = True
                cnv.append((i1, j1))
                if S == 0:
                    S = 6
                elif S == 1:
                    S = 7
                else:
                    S = (S - 2) % 8
                if img[i1][j1] >= 3:
                    img[i1][j1] += 1
                else:
                    img[i1][j1] = 3
                current_point = (i1, j1)
            elif img[i2][j2] == symbol or img[i2][j2] >= 3:
                found = True
                cnv.append((i2, j2))
                if img[i2][j2] >= 3:
                    img[i2][j2] += 1
                else:
                    img[i2][j2] = 3
                current_point = (i2, j2)
            elif img[i3][j3] == symbol or img[i3][j3] >= 3:
                found = True
                cnv.append((i3, j3))
                if img[i3][j3] >= 3:
                    img[i3][j3] += 1
                else:
                    img[i3][j3] = 3
                current_point = (i3, j3)
            else:
                S = (S + 2) % 8
        first = False
    return cnv


def find_contours_teo_bis(data, symbol):
    img = []
    for i in range(len(data)):
        img.append([])
        for j in range(len(data[0])-1):
            if data[i][j] == symbol:
                img[i].append(1)
            else:
                img[i].append(0)
    cnv = []
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] == 1 and j - 1 >= 0 and img[i][j - 1] == 0:
                start_point = (i, j)
                cnv.append(findcontours_teo(start_point, img, 1))
    return cnv


def findcnv(data, i, j, symbol, flags):
    object = []
    q = deque()
    flags.add((i, j))
    object.append([j, i])
    q.append((i, j))
    while q:
        current = q.popleft()
        i_cur = current[0]
        j_cur = current[1]
        if j_cur + 1 < len(data[0]) - 1 and isCNV_old(data, i_cur, j_cur + 1) and not (i_cur, j_cur + 1) in flags:
            object.append([j_cur + 1, i_cur])
            flags.add((i_cur, j_cur + 1))
            q.append((i_cur, j_cur + 1))
            continue
        if i_cur + 1 < len(data) and isCNV_old(data, i_cur + 1, j_cur) and not (i_cur + 1, j_cur) in flags:
            object.append([j_cur, i_cur + 1])
            flags.add((i_cur + 1, j_cur))
            q.append((i_cur + 1, j_cur))
            continue
        if j_cur - 1 >= 0 and isCNV_old(data, i_cur, j_cur - 1) and not (i_cur, j_cur - 1) in flags:
            object.append([j_cur - 1, i_cur])
            flags.add((i_cur, j_cur - 1))
            q.append((i_cur, j_cur - 1))
            continue
        if i_cur - 1 >= 0 and isCNV_old(data, i_cur - 1, j_cur) and not (i_cur - 1, j_cur) in flags:
            object.append([j_cur, i_cur - 1])
            flags.add((i_cur - 1, j_cur))
            q.append((i_cur - 1, j_cur))
            continue
        if i_cur + 1 < len(data) and j_cur - 1 >= 0 and isCNV_old(data, i_cur + 1, j_cur - 1) and not (i_cur + 1,
                                                                                                       j_cur - 1) in flags:
            object.append([j_cur - 1, i_cur + 1])
            flags.add((i_cur + 1, j_cur - 1))
            q.append((i_cur + 1, j_cur - 1))
            continue
        if i_cur + 1 < len(data) and j_cur + 1 < len(data[0]) - 1 and isCNV_old(data, i_cur + 1, j_cur + 1) and not (
                                                                                                                    i_cur + 1,
                                                                                                                    j_cur + 1) in flags:
            object.append([j_cur + 1, i_cur + 1])
            flags.add((i_cur + 1, j_cur + 1))
            q.append((i_cur + 1, j_cur + 1))
            continue
        if i_cur - 1 >= 0 and j_cur - 1 >= 0 and isCNV_old(data, i_cur - 1, j_cur - 1) and not (i_cur - 1,
                                                                                                j_cur - 1) in flags:
            object.append([j_cur - 1, i_cur - 1])
            flags.add((i_cur - 1, j_cur - 1))
            q.append((i_cur - 1, j_cur - 1))
            continue
        if i_cur - 1 >= 0 and j_cur + 1 < len(data[0]) - 1 and isCNV_old(data, i_cur - 1, j_cur + 1) and not (i_cur - 1,
                                                                                                              j_cur + 1) in flags:
            object.append([j_cur + 1, i_cur - 1])
            flags.add((i_cur - 1, j_cur + 1))
            q.append((i_cur - 1, j_cur + 1))
            continue
    return object


def findcntrs():
    cnv = [[], []]
    flags = set()
    for i in range(len(data)):
        for j in range(len(data[0]) - 1):
            if data[i][j] != '0' and isCNV_old(data, i, j) and not (i, j) in flags:
                cnv[int(data[i][j]) - 1].append(findcnv(data, i, j, data[i][j], flags))


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


def notbinsearch(arr, key):
    start = 0
    end = len(arr) - 1
    while start <= end:
        m = (start + end) >> 1
        if key == arr[m]:
            return m
        elif key < arr[m]:
            end = m - 1
        else:
            start = m + 1
    return end


def get_iters(appr, cnv_saved):
    res = []
    cnv = cnv_saved
    for ap in appr:
        for i in range(len(cnv)):
            if ap[0] == cnv[i][0][0] and ap[1] == cnv[i][0][1]:
                res.append([i, ap[0], ap[1]])
                break
    res.sort()
    return res


def calculate_errs(appr, cnv_saved):
    cnv = []
    errs = []
    for p in cnv_saved:
        cnv.append([p[0][0], p[0][1]])
    appr_it = get_iters(appr, cnv)
    tmp_list = [appr_it[k][0] for k in range(len(appr_it))]
    for i in range(len(cnv)):
        tmp = notbinsearch(tmp_list, i)
        if tmp == -1 or tmp == len(appr)-1:
            errs.append(calculate_aev([appr_it[0][1], appr_it[0][2]], [appr_it[len(appr)-1][1], appr_it[len(appr)-1][2]], cnv[i]))
        else:
            errs.append(calculate_aev([appr_it[tmp][1], appr_it[tmp][2]], [appr_it[tmp+1][1], appr_it[tmp+1][2]], cnv[i]))
    return errs


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


def fun(x):
    loss = 0
    x1, y1 = [], []
    for i in range(4):
        points = bounds[i]
        x1.append(points[:, 0])
        y1.append(points[:, 1])
    loss += np.sum((x[0] * x1[0] + x[1] * y1[0] + x[2]) ** 2)
    loss += np.sum((x[1] * x1[1] - x[0] * y1[1] + x[3]) ** 2)
    loss += np.sum((x[0] * x1[2] + x[1] * y1[2] + x[4]) ** 2)
    loss += np.sum((x[1] * x1[3] - x[0] * y1[3] + x[5]) ** 2)
    return loss


def create_rctngl_new(cnv1, appr):
    cnv=[]
    for c in cnv1:
        cnv.append([[c[0][0], sizey - c[0][1]]])
    points = get_iters(appr, cnv)

    points.sort()
    bounds0 = [[], [], [], []]
    for i in range(points[3][0], len(cnv)):
        bounds0[0].append([cnv[i][0][0], cnv[i][0][1]])
    for i in range(points[0][0]):
        bounds0[0].append([cnv[i][0][0], cnv[i][0][1]])
    for i in range(points[0][0]+1, points[1][0]):
        bounds0[1].append([cnv[i][0][0], cnv[i][0][1]])
    for i in range(points[1][0]+1, points[2][0]):
        bounds0[2].append([cnv[i][0][0], cnv[i][0][1]])
    for i in range(points[2][0]+1, points[3][0]):
        bounds0[3].append([cnv[i][0][0], cnv[i][0][1]])

    for i in range(4):
        bounds.append(np.array(bounds0[i]))

    first_line = create_line([points[0][1],points[0][2]], [points[1][1],points[1][2]])
    l0 = [first_line[0], first_line[1], first_line[2], 1, 1, 1]
    # l0 = [1,1,1,1,1,1]
    coef = minimize(fun, l0, method='BFGS')
    lines = []
    lines.append((coef['x'][0], coef['x'][1], coef['x'][2]))
    lines.append((coef['x'][1], - coef['x'][0], coef['x'][3]))
    lines.append((coef['x'][0], coef['x'][1], coef['x'][4]))
    lines.append((coef['x'][1], - coef['x'][0], coef['x'][5]))
    res_points = []
    for i in range(3):
        res_points.append(cross(lines[i], lines[i + 1]))
    res_points.append(cross(lines[0], lines[3]))
    return res_points


def findY(line, x):
    return -(x[0]*line[0] + line[2])/line[1]


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


data = []
i = -1
with open('audi/000001_Audi_Run5.txt', 'r') as f:
    for line in f.readlines():
        if len(line)>1:
            i += 1
            data.append([])
            for c in line:
                if c!=',':
                    data[i].append(c)
sizey = len(data) - 1

cnv = [findcontours_suzuki(data, '1'), findcontours_suzuki(data, '2')]
res_points = []
appr_cnv_first = cv2.approxPolyDP(cnv[0][1], 0.78, True)
res_points.append([])
for c in appr_cnv_first:
    res_points[len(res_points)-1].append([c[0][0], sizey - c[0][1]])
appr_masood_zero = masood(cnv[0][0], False, 20)
res_points.append([])
for c in appr_masood_zero:
    res_points[len(res_points) - 1].append([c[0], sizey - c[1]])
appr_masood_2 = masood(cnv[0][2], True, 0.7)
appr_masood_second = []
for c in appr_masood_2:
    appr_masood_second.append([c[0], sizey - c[1]])

#points 0,3 - big circle arc;1,2 - small circle arc
tmp = [[0,appr_masood_second[0][0],appr_masood_second[0][1]],[0,appr_masood_second[0][0],appr_masood_second[0][1]],
       [0,appr_masood_second[0][0],appr_masood_second[0][1]],[0,appr_masood_second[0][0],appr_masood_second[0][1]]]
for i in range(len(appr_masood_second)):
    if appr_masood_second[i][0] <= tmp[0][1] and appr_masood_second[i][1] > tmp[0][2]:
        tmp[0] = [i,appr_masood_second[i][0], appr_masood_second[i][1]]
    if appr_masood_second[i][0] <= tmp[1][1] and appr_masood_second[i][1] < tmp[1][2]:
        tmp[1] = [i,appr_masood_second[i][0], appr_masood_second[i][1]]
    if appr_masood_second[i][0] > tmp[2][1] and appr_masood_second[i][1] <= tmp[2][2]:
        tmp[2] = [i,appr_masood_second[i][0], appr_masood_second[i][1]]
    if appr_masood_second[i][0] >= tmp[3][1] and appr_masood_second[i][1] >= tmp[3][2]:
        tmp[3] = [i,appr_masood_second[i][0], appr_masood_second[i][1]]


smth = []
for i in range(tmp[0][0]+1, len(appr_masood_second)):
    smth.append(appr_masood_second[i])
for i in range(tmp[1][0]):
    smth.append(appr_masood_second[i])
res_points.append(smth)
smth = []
for i in range(tmp[2][0]+1, tmp[3][0]):
    smth.append(appr_masood_second[i])

line0 = create_line([tmp[0][1],tmp[0][2]], [tmp[1][1],tmp[1][2]])
line1 = create_line([tmp[2][1],tmp[2][2]], [tmp[3][1],tmp[3][2]])
cntr = cross(line0, line1) # centre of circle

angle = find_angle([tmp[2][1], tmp[2][2]], [tmp[3][1], tmp[3][2]])
tmp[2][1] = (tmp[1][2]-cntr[1])*m.cos(angle)+cntr[0]
tmp[2][2] = (tmp[1][2]-cntr[1])*m.sin(angle)+cntr[1]

line1 = create_line([tmp[2][1],tmp[2][2]], [tmp[3][1],tmp[3][2]])
point1 = smth[0]
point2 = smth[len(smth)-1]


while(smth[0][1] <= findY(line1, smth[0])):
    point1 = smth[0]
    smth.pop(0)
while(smth[len(smth)-1][1] <= findY(line1, smth[len(smth)-1])):
    point2 = smth[len(smth)-1]
    smth.pop(len(smth)-1)
line2 = create_line(smth[0], point1)
line3 = create_line(smth[len(smth)-1], point2)

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
for i in range(len(cnv[1])):
    bounds = []
    tmp_appr_2 = []
    tmp_appr = masood(cnv[1][i], False, 4)
    for c in tmp_appr:
        tmp_appr_2.append([c[0], sizey - c[1]])

    coolpoints = create_rctngl_new(cnv[1][i], tmp_appr_2)
    res_points.append([])
    for j in range(len(coolpoints)):
        res_points[len(res_points)-1].append([coolpoints[j][0], coolpoints[j][1]])

model = gmsh.model
factory = model.geo

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)

model.add("rotor")
lc = 1e-2

factory.addPoint(cntr[0], cntr[1], 0, lc, 1)
for i in range(4):
    factory.addPoint(tmp[i][1], tmp[i][2], 0, lc, i+2)
factory.addLine(2, 3, 1)
factory.addCircleArc(3, 1, 4, 2)
factory.addLine(4, 5, 3)
factory.addCircleArc(5, 1, 2, 4)

point_cnt = 5
for i in range(len(res_points)):
    for j in range(len(res_points[i])):
        point_cnt += 1
        factory.addPoint(res_points[i][j][0], res_points[i][j][1], 0, lc, point_cnt)
    start = point_cnt - len(res_points[i]) + 1
    for j in range(len(res_points[i])-1):
        factory.addLine(start+j, start+j+1)
    factory.addLine(point_cnt, start)

factory.synchronize()
model.mesh.generate(2)

gmsh.write("audi/1.geo_unrolled")
gmsh.finalize()


#подсчет ошибок
# res_dp = calculate_errs(appr_cnv, cnv[0][0])
# res_it_del = calculate_errs(appr_iterate_deleting, cnv[0][0])
# res_mas = calculate_errs(appr_masood, cnv[0][0])
#
# print('DP')
# print(len(appr_cnv))
# # print(res_dp)
# print('MaxErr =', max(res_dp))
# print('ISE =', sum(res_dp))

# print('it_del')
# print(len(appr_iterate_deleting))
# # print(res_it_del)
# print('MaxErr =', max(res_it_del))
# print('ISE =', sum(res_it_del))
#
# print('masood')
# print(len(appr_masood))
# # print(res_mas)
# print('MaxErr =', max(res_mas))
# print('ISE =', sum(res_mas))