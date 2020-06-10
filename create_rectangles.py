import numpy as np
from scipy.optimize import minimize


def create_line(p1, p2):
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]
    A = y2 - y1
    B = x1 - x2
    C = (x2 - x1)*y1 - (y2 - y1)*x1
    return [A, B, C]


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


def fun(x, bounds):
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


def create_rctngl_new(cnv1, appr, sizey, bounds):
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
    coef = minimize(fun, l0, bounds, method='BFGS')
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


def create_rectangles(cnv, res_points, sizey):
    for i in range(len(cnv[1])):
        bounds = []
        tmp_appr_2 = []
        tmp_appr = masood(cnv[1][i], False, 4)
        for c in tmp_appr:
            tmp_appr_2.append([c[0], sizey - c[1]])
        coolpoints = create_rctngl_new(cnv[1][i], tmp_appr_2, sizey, bounds)
        res_points.append([])
        for j in range(len(coolpoints)):
            res_points[len(res_points)-1].append([coolpoints[j][0], coolpoints[j][1]])
