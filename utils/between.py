import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import pandas as pd
import time

from scipy.ndimage import gaussian_filter1d
from matplotlib.font_manager import FontProperties

flag = False

def readcsv(files, smooth=False):
    datas = []
    for file in files:
        data = pd.read_csv(file)
        x = data['Step'].to_numpy()
        y = data['Value'].to_numpy()
        datas.append(y)

    datas = np.array(datas)   
    mean = datas.mean(axis=0)
    std = datas.std(axis=0)

    r1 = mean + std
    r2 = mean - std

    if smooth:
        mean = gaussian_filter1d(mean, sigma=2)
        r1 = gaussian_filter1d(r1, sigma=2)
        r2 = gaussian_filter1d(r2, sigma=2)

    return x, mean, r1, r2
 
# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
config = {
            "font.family":'Times New Roman',  # 设置字体类型
                # "font.size": 14,
                #     "mathtext.fontset":'stix',
                }

plt.rcParams.update(config)
# plt.figure(figsize=(15,5))

files = ['csv/vrun1.csv', 'csv/vrun3.csv', 'csv/vrun4.csv', 'csv/vrun5.csv']

# x, r, r1, r2 = readcsv(files, True)
x, r, r1, r2 = readcsv(files)
x, r, r1, r2 = x[:120], r[:120], r1[:120], r2[:120]
plt.plot(x, r, color='orangered', label='VD3QN')
plt.fill_between(x, r1, r2, color='orangered', alpha=0.1)
plt.legend(['VD3QN'])
if flag:
    plt.ylim(-25, 15)
    plt.legend()
    plt.savefig('VD3QN_Mean.svg')
    plt.cla()

files = ['csv/drun1.csv', 'csv/drun2.csv', 'csv/drun3.csv', 'csv/drun4.csv', 'csv/drun5.csv']
# x, r, r1, r2 = readcsv(files, True)
x, r, r1, r2 = readcsv(files)
x, r, r1, r2 = x[:120], r[:120], r1[:120], r2[:120]
plt.plot(x, r, color='steelblue', label='D3QN')
plt.fill_between(x, r1, r2, color='steelblue', alpha=0.1)
if flag:
    plt.ylim(-25, 15)
    plt.legend()
    plt.savefig('D3QN_Mean.svg')
    plt.cla()

files = ['csv/run_v1.csv', 'csv/run_v2.csv', 'csv/run_v3.csv', 'csv/run_v4.csv', 'csv/run_v5.csv']
# x, r, r1, r2 = readcsv(files, True)
x, r, r1, r2 = readcsv(files)
x, r, r1, r2 = x[:120], r[:120], r1[:120], r2[:120]
plt.plot(x, r, color='green', label='VD3QN-')
plt.fill_between(x, r1, r2, color='green', alpha=0.1)
if flag:
    plt.ylim(-25, 15)
    plt.legend()
    plt.savefig('D3QN_V_Mean.svg')
    plt.cla()

if not flag:
    plt.legend()
    plt.savefig('compare.svg')
