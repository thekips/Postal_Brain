import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from scipy.ndimage import gaussian_filter1d

'''读取csv文件'''
def readcsv(files, smooth=False):
    data = pd.read_csv(files)
    x = data['Step'].to_numpy()
    y = data['Value'].to_numpy()

    if smooth:
        y = gaussian_filter1d(y, sigma=5)

    return x ,y
 
# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
config = {
            "font.family":'Times New Roman',  # 设置字体类型
                "font.size": 14,
                #     "mathtext.fontset":'stix',
                }

plt.rcParams.update(config)
plt.figure(figsize=(15,5))

sub1 = plt.subplot(1,2,2)
ax=plt.gca();#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

x1, y1 = readcsv("csv/Mean_Reward_VD3QN.csv", True)
plt.plot(x1[:130], y1[:130], color='orangered',label='VD3QN')

x2, y2 = readcsv("csv/Mean_Reward_D3QN.csv", True)
plt.plot(x2[:130], y2[:130], color='steelblue', label='D3QN')
 
x3, y3 = readcsv("csv/Mean_Reward__VD3QN.csv", True)
plt.plot(x3[:130], y3[:130], color='green',label='VD3QN-')

plt.xlabel('Episodes', fontsize=16)
plt.ylabel('Rewards', fontsize=16)
plt.legend(fontsize=12)
 
sub2 = plt.subplot(1,2,1)
ax=plt.gca();#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

x4, y4 = readcsv("csv/vloss.csv", True)
plt.plot(x4[:633], y4[:633], color='orangered',label='VD3QN')

x5, y5 = readcsv("csv/dloss.csv", True)
plt.plot(x5[:633], y5[:633], color='steelblue', label='D3QN')
 
x6, y6 = readcsv("csv/loss_v.csv", True)
plt.plot(x6[:633], y6[:633], color='green',label='VD3QN-')
 
plt.xlabel('Steps', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.legend(fontsize=12)

plt.savefig('ablation.svg')

plt.show()

