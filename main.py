from find_contours import findcontours_suzuki
from approx import approx
from create_rectangles import create_rectangles
from create_model import create_model


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

#find contours
cnv = [findcontours_suzuki(data, '1'), findcontours_suzuki(data, '2')]

#approximate contours
res_points = []
outer_points, cntr = approx(cnv, res_points, sizey)

#create_rectangles
create_rectangles(cnv, res_points, sizey)

#create .geo file using gmsh
name = "audi/1.geo_unrolled"
create_model(res_points,outer_points,cntr,name)
