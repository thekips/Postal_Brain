import numpy as np
import pandas as pd

def read_csv(path, low_memory=False):
    '''
    读取csv文件

    参数：
    path -- 需要读取文件的路径
    low_memory -- 是否以低内存形式读取：True代表是，False代表否

    返回：
    包含csv文件的dataframe对象
    '''
    try:
        return pd.read_csv(path,encoding='gb18030',low_memory=low_memory)
    except:
        return pd.read_csv(path,low_memory=low_memory)

def read_cx(path):
    '''
    读取csv或者excel文件
    '''
    try:
        return read_csv(path)
    except:
        return pd.read_excel(path)

def latlng2_manhattan_distance(loc1, loc2):
    '''
    计算loc1与loc2之间的曼哈顿距离

    参数：
    loc1 -- 含经纬度的array
    loc2 -- 含经纬度的array

    返回：
    c -- loc1与loc2之间的曼哈顿距离
    '''
    lat_lon_1 = np.radians(loc1)
    lat_lon_2 = np.radians(loc2)
    d_lat_lon = np.abs(lat_lon_1- lat_lon_2)
    
    r = 6373.0
    a_lat_lon = np.sin(d_lat_lon / 2.0) **2
    c = 2 * np.arctan2(np.sqrt(a_lat_lon), np.sqrt(1 - a_lat_lon))
    c = r * c
    c = c.reshape(-1,2)
    c = np.sum(c, axis=1)
    
    return c