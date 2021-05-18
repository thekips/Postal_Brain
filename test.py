import numpy as np

def location_to_manhattan(loc1, loc2):
    '''
    计算loc1与loc2之间的曼哈顿距离

    参数：
    loc1: 含经纬度的array
    loc2: 含经纬度的array

    返回：
    c: loc1与loc2之间的曼哈顿距离
    '''
    d_lat_lon = np.abs(np.radians(loc1) - np.radians(loc2))
    
    r = 6373.0
    a_lat_lon = np.sin(d_lat_lon / 2.0) **2
    c = 2 * np.arctan2(np.sqrt(a_lat_lon), np.sqrt(1 - a_lat_lon))
    c = r * c
    c = c.reshape(-1,2)
    c = np.sum(c, axis=1)
    
    return c if len(c) > 1 else c[0]

x = [121, 134]
y = [(121, 135)]
z = location_to_manhattan(x, y)
print(z)