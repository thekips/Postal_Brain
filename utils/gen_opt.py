import json
import os
import threading
import numpy as np
import pandas as pd
from concorde.tsp import TSPSolver
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
def draw_route(x, y,tour_data):
    
    x_coor=[x[tour_data[i]] for i in range(0,len(x))]
    y_coor =[y[tour_data[i]] for i in range(0,len(y))]
    plt.step(x_coor, y_coor,'-o')
    plt.plot(x_coor[0],y_coor[0],'g^',markersize=10)
    plt.show()

def getCenterPoint(data, num):
    x_min, x_max = data['lng'].min(), data['lng'].max()
    y_min, y_max = data['lat'].min(), data['lat'].max()
    x_len = x_max - x_min
    y_len = y_max - y_min

    print(x_min, x_max)
    print(y_min, y_max)

    point_x = x_min + [x_len * (1 / 2 + i) / num for i in range(num)]
    point_y = y_min + [y_len * (1 / 2 + i) / num for i in range(num)]

    return point_x, point_y

def calTime(start_point):
    xx = data['lat'].to_numpy()
    yy = data['lng'].to_numpy()
    np.insert(xx, 0, start_point[0])
    np.insert(yy, 0, start_point[1])
    xx = xx * 10
    yy = yy * 10
    # xx = xx * 1e3
    # yy = yy * 1e3
    # print(xx[0], yy[0])

    solver = TSPSolver.from_data(xs=xx, ys=yy, norm='MAN_2D')
    res = solver.solve(verbose=False)

    with lock:
        writeRes(str(start_point), res.optimal_value, path)

def writeRes(key, value, path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    data[key] = value
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


data = pd.read_csv('data/env.csv')
dep = 52900009
data = data.loc[data['投递机构'] == dep]
data = data.iloc[:10000]
print("thekips: len of data is %d." % len(data))

#%%
point_x, point_y = getCenterPoint(data, 100)
path = 'opt_value.json'

points = []
for x in point_x:
    for y in point_y:
        points.append((x,y))

lock = threading.Lock()

with ThreadPoolExecutor(8) as executor:
    executor.map(calTime, points) 
