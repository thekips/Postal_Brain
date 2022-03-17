#%%
import json
import os
import threading
import numpy as np
import pandas as pd
from concorde.tsp import TSPSolver
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt


def location_to_manhattan(loc1, loc2):
    """
    caculate distance between loc1 and loc2.

    Args：
        loc1: A array, list or tuple include location pair.
        loc2: A array, list or tuple include location pair.

    Returns：
        dist: manhattan distance between loc1 and loc2.
    """
    abs = np.abs(np.array(loc1) - np.array(loc2))
    return np.sum(abs)


def draw_route(x, y, tour_data):

    x_coor = [x[tour_data[i]] for i in range(0, len(x))]
    y_coor = [y[tour_data[i]] for i in range(0, len(y))]
    plt.step(x_coor, y_coor, "-o")
    plt.plot(x_coor[0], y_coor[0], "g^", markersize=10)
    plt.show()


def getCenterPoint(data, num):
    x_min, x_max = data["lat"].min(), data["lat"].max()
    y_min, y_max = data["lng"].min(), data["lng"].max()
    x_len = x_max - x_min
    y_len = y_max - y_min

    print(x_min, x_max)
    print(y_min, y_max)

    point_x = x_min + [x_len * (1 / 2 + i) / num for i in range(num)]
    point_y = y_min + [y_len * (1 / 2 + i) / num for i in range(num)]

    return point_x, point_y


def calTime(start_point, precision=1000, isplot=False):

    xx = data["lat"].to_numpy()
    yy = data["lng"].to_numpy()
    xx = np.insert(xx, 0, start_point[0])
    yy = np.insert(yy, 0, start_point[1])
    xx = xx * precision
    yy = yy * precision
    print(xx[0], yy[0])
    # xx = xx * 1e3
    # yy = yy * 1e3
    # print(xx[0], yy[0])

    solver = TSPSolver.from_data(xs=xx, ys=yy, norm="MAN_2D")
    res = solver.solve(verbose=False)
    x = [xx[i] / precision for i in res.tour]
    x.append(start_point[0])
    y = [yy[i] / precision for i in res.tour]
    y.append(start_point[1])
    if isplot == True:
        import matplotlib.pyplot as plt

        plt.plot(x, y, "o")
        plt.plot(x[0], y[0], "r^")
        plt.axis("off")
    
    solution = np.array([*zip(x, y)])
    print(solution)
    distance = location_to_manhattan(solution[:-1], solution[1:])
    print(distance)
    
    with lock:
        writeRes(str(start_point), distance, path)


def writeRes(key, value, path):
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[key] = value
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


data = pd.read_csv("data/env.csv")
data = data.drop_duplicates(["lng", "lat"], keep="first")
data = data.iloc[:1000]
print("Len of data is %d." % len(data))

#%%
import time

a = time.time()
point = (data["lat"].mean(), data["lng"].mean())
print("point is:", point)
tour = calTime(point, 1000, True)
print(time.time() - a)

#%%
point = (22.30879190138971, 112.37251670934572)
calTime(point, 1000, True)

#%%
import json
with open('dist/100_opt_value.json', 'r') as f:
    temp = json.load(f)
key = min(temp, key=temp.get)
print(key, temp[key])
#%%
point_x, point_y = getCenterPoint(data, 100)
path = "opt_value.json"

points = []
for x in point_x:
    for y in point_y:
        points.append((x, y))
#%%
print(points[6700])
tour = calTime(points[6770], 1000, True)
#%%
point_x, point_y = getCenterPoint(data, 100)

points = []
for x in point_x:
    for y in point_y:
        points.append((x, y))
print("point is:", point)
point = points[5050]
tour = calTime(point, 1000, True)

#%%
grid = 100
xx = data["lat"].to_numpy()
xx = np.insert(xx, 0, point[0])
xx = grid * (xx - xx.min()) / (xx.max() - xx.min())
yy = data["lng"].to_numpy()
yy = np.insert(yy, 0, point[1])
yy = grid * (yy - yy.min()) / (yy.max() - yy.min())
xx = [xx[i] for i in tour]
yy = [yy[i] for i in tour]

import seaborn as sns

fig = plt.figure()
plt.cla()
plt.plot(xx, yy, "-o")
plt.plot(xx[0], yy[0], "r^")
# plt.axis('off')
fig.canvas.draw()
plt.show(block=False)
plt.pause(0.05)
#%%
point_x, point_y = getCenterPoint(data, 10)
path = "opt_value.json"

points = []
for x in point_x:
    for y in point_y:
        points.append((x, y))

lock = threading.Lock()

with ThreadPoolExecutor(8) as executor:
    executor.map(calTime, points)
