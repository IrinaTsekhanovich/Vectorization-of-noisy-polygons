import gmsh

def create_model(res_points, outer_points, cntr, name):
    model = gmsh.model
    factory = model.geo

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    model.add("rotor")
    lc = 1e-2

    factory.addPoint(cntr[0], cntr[1], 0, lc, 1)
    for i in range(4):
        factory.addPoint(outer_points[i][0], outer_points[i][1], 0, lc, i + 2)
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

    gmsh.write(name)
    gmsh.finalize()
