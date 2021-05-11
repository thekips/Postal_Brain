import arrow
import csv
import numpy as np
import pandas as pd
import os
import re
from datetime import datetime
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class Utils:

    def __init__(self, department, select, partition, weilan, weilans):
        self.department = department # 需要处理的机构部门文件路径
        self.select = select # 机构部门经纬度文件的路径
        self.partition = partition # CBD等区域划分文件的路径
        self.table1 = readCX(weilan) # 机构围栏表
        self.table2 = readCX(weilans) # 机构围栏坐标点表

    def subDepartment(self, data):
        '''
        将不需要处理的部门的数据排除
        '''
        departments = readCX(self.department)
        departments = departments['机构代码'].tolist()
        bools = [x in departments for x in data['投递机构'].values]
        
        for i,x in enumerate(bools):
            if x == False:
                print(i)
            
        return data.loc[bools]

    def subAdmin(self, data):
        '''
        删除机构管理员的记录
        '''
        index = data[data['投递员'].str.contains('机构管理员')].index
        return data.drop(index)
    
    def subHoliday(self, data, years):
        '''
        将双休日的工作数据排除

        参数：
        data -- 需要处理的数据
        years -- 包含需要排除的年份的列表

        返回：
        排除双休日之后的工作数据
        '''
        all_date_list = []
        for year in years:
            all_date_list += getAllDayPerYear(year)
        
        week_days = [] # weekdays为周末的日期列表
        for date in all_date_list:
            if pd.to_datetime(date).isoweekday() > 5:
                week_days.append(date)
        week_days = set(week_days)

        try:
            bools = [str(x[:10]) not in week_days for x in data['投递日期'].values]
        except:
            bools = [str(x.date()) not in week_days for x in data['投递日期'].values] 
            
        return data.loc[bools]
    
    def subWeilan(self, data):
        '''
        将电子围栏围栏外边的工作数据排除

        参数：
        data -- 需要处理的数据

        返回：
        排除电子围栏外边之后的工作数据
        '''
        check = readCX(self.department)['机构代码'].values
        for department in check:
            batch = data.loc[data['投递机构']==int(department)]
            weilans = self.table1[self.table1['机构代码'] == int(department)]['ID主键'].values
            #有电子围栏才进行判断
            if len(weilans) > 0:
                bools = np.zeros(len(batch),dtype=bool) #设置为全False
                for ID in weilans:
                    weilan = self.table2[self.table2['GIS_ID']==ID][['V_LONGITUDE','V_LATITUDE']].values
                    bools += isInPoly(batch[['lng', 'lat']].values, weilan)
                data = data.drop(batch[~bools].index)            
        return data

    def subStreet(self, x):
        '''
        对地址进行后缀删除
        如“小谷围街道132号”转换为“小谷围街道”
        '''
        try:
            return re.search(r'[\u4e00-\u9fa5]+(?=[′ⅠⅡˉ―一－–A-Za-z\s\d\*\.\-]*.$)', x).group()
        except:
            return ''

    def genStreet(self, data):
        '''
        从data中提取每个街道名存放至sets中，并将街道名与经纬度最值存放至dicts中

        参数：
        data -- 需要提取的数据

        返回：
        sets -- 街道名称，set
        dicts -- 街道的经纬度最值，dataframe
        '''
        df = data[data['投递地址'].str.contains('[\d]+号$', na=False)][['投递地址','lng','lat']] # 包含号的记录
        df = df.drop(df[df['投递地址'].str.contains('[\*\.\s]')].index) # 筛除特殊字符的记录
        df['街道'] = df['投递地址'].apply(self.subStreet) # 将该记录的xx号后缀全部去除
        df = df.drop(df[df['街道']==''].index) # 处理过程中无法正常去除后缀的记录筛除
        df = df.drop(df[df['街道'].isnull()].index) # 空白值筛除
        df = df.drop(df[df['街道'].str.len()<3].index) # 长度小于3筛除

        dicts = df.groupby('街道').agg({'lng':[np.max, np.min], 'lat':[np.max, np.min]})
        sets = set(df.index.values)

        return dicts, sets

    def subAbnormal(self, data):
        #TODO
        #保留在江门市之内的地址以及空白地址
        index = data.loc[data['投递地址'].str.contains('江门|台山|开平|恩平|鹤山|蓬江区|江海区|新会区',na=False)].index
        index = index.append(data.loc[data['投递地址'].isnull()].index).sort_values()
        
        return data.loc[index]

    def subLngLat(self, data):
        #TODO
        #筛除经纬度超出江门市的地址
        index = data.loc[(data['lng']>113.25) | (data['lng']<112) | (data['lat']>22.9) | (data['lat']<21.5)].index
        
        return data.drop(index)

    def makeType(self, data):
        '''
        新增一个分类，用于判断邮件类型
        快包 -- 1.2
        标快 -- 1.5
        增值 -- 2
        高端 -- 4
        '''
        data['分类'] = 1.2
        data.loc[data['业务种类'].str.contains(r'快包|其他', na=False),'分类'] = 1.2
        data.loc[data['业务种类'].str.contains(r'标快|标准EMS|国际', na=False),'分类'] = 1.5
        data.loc[data['业务种类'].str.contains(r'法院', na=False),'分类'] = 2
        data.loc[data['业务种类'].str.contains(r'高端政务', na=False),'分类'] = 4

        data.loc[data['邮件号'].str.contains(r'^1.*8[0-9]$', na=False),'分类'] = 2 #增值
        data.loc[data['邮件号'].str.contains(r'^1.*9[2-4]$', na=False),'分类'] = 2 #法院
        
        return data

    def makeWeight(self, data):
        '''
        设定重量系数
        '''
        data.loc[data['重量']<5,'重量'] = 1
        data.loc[data['重量'].isnull(),'重量'] = 1 #空值默认小于5
        data.loc[data['重量']>=5,'重量'] = 1.2
        
        return data

    def makeHard(self, data):
        '''
        设定地址类型难度系数
        住宅 -- 1
        CBD -- 1.15
        城中村 -- 1.4
        '''
        partitions = readCX(self.partition)
        partitions['lng'] = partitions['中心坐标'].map(lambda x: float(x.split(',')[0]))
        partitions['lat'] = partitions['中心坐标'].map(lambda x: float(x.split(',')[1]))
        partitions['辐射半径(m)'] = partitions['辐射半径(m)'].map(lambda x: x / 1000)

        data['type'] = 1
        for i in range(len(partitions)):
            if partitions.loc[i,:][6] == 'CBD':
                weight = 1.15
            if partitions.loc[i,:][6] == '城中村':
                weight = 1.4
            temp = np.array(data[['lng','lat']])
            core = np.array(partitions.loc[i,:][7:9]).astype(np.float32)
            dist = latlng2_manhattan_distance(core, temp)
            #选择在radius范围之内的记录
            radius = partitions.loc[i,:][5]
            index =data.loc [dist < radius].index
            if len(index) > 0:
               data.loc[dist < radius, 'type'] = weight

        return data

    def makeSpecial(self, data):
        '''
        揽收数据的揽收地址有自己的地址，特殊设置为机构经纬度
        '''
        departments = readCX(self.select)
        departments = departments[['机构代码','lat','lng']]
        temp = data[data['投递地址'].str.contains('寄递事业部',na=False)]

        # 营业部将经纬度特殊处理为机构经纬度，即后续dist计算为0
        for i in range(len(departments)):
            index = temp[temp['投递机构'] == departments.loc[i,:][0]].index
            data.loc[index, 'lat'] = departments.loc[i,:][1]
            data.loc[index, 'lng'] = departments.loc[i,:][2]

        return data

    def calDistance(self, data):
        '''
        以self.select中的机构经纬度为依据，计算data中每条记录到记录对应机构的曼哈顿距离,存至'dist'列

        参数：
        data -- 需要处理的数据

        返回：
        计算完每条记录至记录对应机构的曼哈顿距离的data
        '''
        departments = readCX(self.select)
        departments = departments[['机构代码','lat','lng']]

        data['dist'] = ''
        for i in range(len(departments)):
            temp = data.loc[data['投递机构'] == departments.loc[i,:][0], ['lat','lng']]
            dist = latlng2_manhattan_distance(np.array(departments.loc[i,:][1:3]), np.array(temp))
            data.loc[data['投递机构'] == departments.loc[i,:][0], 'dist'] = dist
        
        return data

def makeDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def readCsv(path, low_memory=False):
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

def readCX(path):
    '''
    读取csv或者excel文件
    '''
    try:
        return readCsv(path)
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

def isLeapYear(years):
    '''
    通过判断闰年，获取年份years下一年的总天数

    参数：
    years -- 年份

    返回：
    days_sum -- 一年的总天数
    '''
    # 断言：年份不为整数时，抛出异常。
    assert isinstance(years, int), "请输入整数年，如 2020"

    if ((years % 4 == 0 and years % 100 != 0) or (years % 400 == 0)):  # 判断是否是闰年
        # print(years, "是闰年")
        days_sum = 366
        return days_sum
    else:
        # print(years, '不是闰年')
        days_sum = 365
        return days_sum

def getAllDayPerYear(years):
    '''
    获取一年的所有日期

    参数：
    years -- 年份

    返回：
    all_date_list -- 一年中全部日期列表
    '''
    start_date = '%s-1-1' % years
    a = 0
    all_date_list = []
    days_sum = isLeapYear(int(years))
    while a < days_sum:
        b = arrow.get(start_date).shift(days=a).format("YYYY-MM-DD")
        a += 1
        all_date_list.append(b)
    
    return all_date_list

def isInPoly(points, poly):
    '''
    判断所有输入的点是否在poly内，返回一个包含布尔值类型的np.array

    参数：
    point -- 需要测试的单个点
    poly -- 用于测试的多边形array

    返回：
    包含布尔值类型的np.array
    '''
    poly = Polygon(poly) # 生成Polygon实例
    return np.apply_along_axis(lambda x: poly.contains(Point(x)), 1, points)