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
    abs = np.abs(np.array(loc1) - np.array(loc2))
    return np.sum(abs)


def draw_route(x, y,tour_data):
    
    x_coor=[x[tour_data[i]] for i in range(0,len(x))]
    y_coor =[y[tour_data[i]] for i in range(0,len(y))]
    plt.step(x_coor, y_coor,'-o')
    plt.plot(x_coor[0],y_coor[0],'g^',markersize=10)
    plt.show()

def getCenterPoint(data, num):
    x_min, x_max = data['lat'].min(), data['lat'].max()
    y_min, y_max = data['lng'].min(), data['lng'].max()
    x_len = x_max - x_min
    y_len = y_max - y_min

    print(x_min, x_max)
    print(y_min, y_max)

    point_x = x_min + [x_len * (1 / 2 + i) / num for i in range(num)]
    point_y = y_min + [y_len * (1 / 2 + i) / num for i in range(num)]

    return point_x, point_y

def calTime(start_point, isplot=False):

    xx = data['lat'].to_numpy()
    yy = data['lng'].to_numpy()
    xx = np.insert(xx, 0, start_point[0])
    yy = np.insert(yy, 0, start_point[1])
    xx = xx * 1000
    yy = yy * 1000
    # xx = xx * 1e3
    # yy = yy * 1e3
    # print(xx[0], yy[0])

    solver = TSPSolver.from_data(xs=xx, ys=yy, norm='MAN_2D')
    res = solver.solve(verbose=False)
    x = [xx[i] / 1000 for i in res.tour]
    x.append(start_point[0])
    y = [yy[i] / 1000 for i in res.tour]
    y.append(start_point[1])

    solution = np.array([*zip(x,y)])
    distance = location_to_manhattan(solution[:-1], solution[1:])
    with lock:
        p1 = (start_point[0], start_point[1])
        writeRes(str(p1), distance, path)
        p2 = (start_point[2], start_point[3])
        writeRes(str(p2), distance, 'true_' + path)

def writeRes(key, value, path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    data[key] = value
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


data = pd.read_csv('../data/env.csv')
data = data.drop_duplicates(['lng', 'lat'], keep='first')
data = data.iloc[:1000]
print("thekips: len of data is %d." % len(data))

#%%
point_x, point_y = getCenterPoint(data, 10)
path = 'opt_value.json'

points = []
for i in range(len(point_x)):
    for j in range(len(point_y)):
        points.append((point_x[i], point_y[j], i, j))
print('len of points is', len(points))

lock = threading.Lock()

with ThreadPoolExecutor(32) as executor:
    executor.map(calTime, points) 
