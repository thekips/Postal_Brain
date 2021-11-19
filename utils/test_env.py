#%%
from datetime import time
import os
import sys
sys.path.append(os.getcwd())
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

CWD = os.path.dirname(os.getcwd()) + '/'

def read_csv(path, low_memory=False) -> DataFrame:
    '''
    read a csv file with encoding=gb18030.

    Args：
        path: the path of csv file.
        low_memory: whether read file in low_memory mode.

    Returns:
        A dataframe object which include the content of file with the given path.
    '''
    try:
        return pd.read_csv(path,encoding='gb18030',low_memory=low_memory)
    except:
        return pd.read_csv(path,low_memory=low_memory)

def read_cx(path) -> DataFrame:
    '''
    Read a csv file or excel file like '.xlsx', '.xls'.

    Args:
        path: the path of csv file or excel file.

    Returns:
        A dataframe object which include the content of file with the given path.
    '''
    try:
        return read_csv(path)
    except:
        return pd.read_excel(path, engine='openpyxl')

def location_to_manhattan(loc1, loc2):
    '''
    caculate distance between loc1 and loc2.

    Args：
        loc1: A array, list or tuple include location pair.
        loc2: A array, list or tuple include location pair.

    Returns：
        dist: manhattan distance between loc1 and loc2.
    '''
    d_lat_lon = np.abs(np.radians(loc1) - np.radians(loc2))
    
    r = 6373.0
    a_lat_lon = np.sin(d_lat_lon / 2.0) **2
    distance = 2 * np.arctan2(np.sqrt(a_lat_lon), np.sqrt(1 - a_lat_lon))
    distance = r * distance
    distance = distance.reshape(-1,2)
    distance = np.sum(distance, axis=1)
    
    return distance if len(distance) > 1 else distance[0]

#%%
import json
import numpy as np
import pandas as pd
from concorde.tsp import TSPSolver

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

def calTime(data, start_point):
    xx = data['lat'].values.tolist()
    yy = data['lng'].values.tolist()
    xx.insert(0, start_point[0])
    yy.insert(0, start_point[1])

    solver = TSPSolver.from_data(xx, yy, 'MAN_2D')
    res = solver.solve(verbose=False)

    return res.optimal_value

def writeRes(key, value, path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    data[key] = value
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


#%%
# Get data to run
data = pd.read_csv('data/lanshou_.csv', encoding='gb18030')
#%%
dep = 52910017
data = data.loc[data['投递机构'] == dep]
print("thekips: len of data is %d." % len(data))
x = data.groupby(['lng','lat']).sum()
index = x.index.to_numpy()
weight = x['重量'].to_numpy()

#%%
point_x, point_y = getCenterPoint(data, 1000)
from matplotlib import pyplot as plt
path = 'opt_value.json'

for x in point_x:
    for y in point_y:
        print("The location to solve is:", x, y)
        optimal_value = calTime(data, (x, y))
        writeRes((x,y), optimal_value, path)