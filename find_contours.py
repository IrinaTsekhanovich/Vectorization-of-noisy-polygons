import numpy as np
import cv2
from collections import deque



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


def findcntrs(data):
    cnv = [[], []]
    flags = set()
    for i in range(len(data)):
        for j in range(len(data[0]) - 1):
            if data[i][j] != '0' and isCNV_old(data, i, j) and not (i, j) in flags:
                cnv[int(data[i][j]) - 1].append(findcnv(data, i, j, data[i][j], flags))


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

