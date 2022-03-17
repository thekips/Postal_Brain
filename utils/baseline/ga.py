#%%
import random
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import math
import os
from concorde.tsp import TSPSolver
from torch.utils.tensorboard import SummaryWriter
import time

writer = SummaryWriter(log_dir="ga/cost" + str(int(time.time())))

num_population = 50  # 随机生成的初始解的总数
num_select = 35  # 保留的解的个数
num_cross = 10  # 交叉解的个数
num_mutation = 5  # 变异解的个数

path = os.getcwd() + "/../data/env.csv"
data = pd.read_csv(path)
data = data.drop_duplicates(subset=["lat", "lng"], keep="first")
data = data.iloc[:1000]
print("Len of data is:", len(data))
lng = data["lng"].to_numpy()
lat = data["lat"].to_numpy()
max_lng, min_lng = lng.max(), lng.min()
max_lat, min_lat = lat.max(), lat.min()


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


# 随机生成初始解[[],[],[]...]
def generate_initial():
    initial = []
    for i in range(num_population):
        x = random.random() * (max_lng - min_lng) + min_lng
        y = random.random() * (max_lat - min_lat) + min_lat
        initial.append((x, y))

    return initial


# 对称矩阵，两个城市之间的距离
def optimal_value(start_point, precision=1000, isplot=False):
    xx = data["lat"].to_numpy()
    yy = data["lng"].to_numpy()
    xx = np.insert(xx, 0, start_point[0])
    yy = np.insert(yy, 0, start_point[1])
    xx = xx * precision
    yy = yy * precision
    # xx = xx * 1e3
    # yy = yy * 1e3
    # print(xx[0], yy[0])

    solver = TSPSolver.from_data(xs=xx, ys=yy, norm="MAN_2D")
    res = solver.solve(verbose=False)
    x = [xx[i] / precision for i in res.tour]
    x.append(start_point[0])
    y = [yy[i] / precision for i in res.tour]
    y.append(start_point[1])

    solution = np.array([*zip(x, y)])
    print(solution)
    distance = location_to_manhattan(solution[:-1], solution[1:])
    
    return distance


# 目标函数计算,适应度计算，中间计算。适应度为1/总距离*10000
def cal_adaptation(population):
    adaptation = []

    for i in range(num_population):
        adaptation.append(optimal_value(population[i]))

    return adaptation


def selection(adaptation, population):
    index = np.argsort(adaptation)[ :num_select]
    print([adaptation[x] for x in index])
    reserve = [population[x] for x in index]
    print('len reserve', len(reserve))

    return reserve


# 随机选择保留下来的70中的25个进行交叉
def cross(population):
    child = []
    for i in range(num_cross):
        parent1 = random.randint(0, len(population) - 1)  # 选择对那个解交叉
        parent2 = random.randint(0, len(population) - 1)  # 选择对那个解交叉
        lat = population[parent1][0] + population[parent2][0]
        lng = population[parent1][1] + population[parent2][1]

        child.append((lat, lng))
    print('len after cross', len(population) + len(child))

    return population + child


# 随机选择那95中的5个进行变异
def mutation(population):
    freak = []
    for i in range(int(num_mutation / 2)):
        parent1 = random.randint(0, len(population) - 1)  # 选择对那个解交叉
        parent2 = random.randint(0, len(population) - 1)  # 选择对那个解交叉
        lat = random.random() * (max_lat - min_lat) + min_lat
        lng = random.random() * (max_lng - min_lng) + min_lng

        freak.append((lat, population[parent1][1]))
        freak.append((population[parent2][0], lng))
    
    print(len(population) + len(freak))

    return population + freak


population = generate_initial()
cnt = 1
while True:
    print('The %d generation, have %d population.' % (cnt, len(population)))
    adaptation = cal_adaptation(population)
    min_cost = min(adaptation)
    mean_cost = np.mean(adaptation)
    writer.add_scalar('min_cost', min_cost)
    writer.add_scalar('mean_cost', mean_cost)
    print('min_cost is %s, mean_cost is %s' % (min_cost, mean_cost))
    
    population = selection(adaptation, population)
    population = cross(population)
    population = mutation(population)

# %%
