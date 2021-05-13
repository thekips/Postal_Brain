import csv
import numpy as np
import pandas as pd
import time

from utils import makeDir, readCX, latlng2_manhattan_distance

class Cost:
    def __init__(self, velocity, ratio):
        self.velocity = velocity
        self.ratio = ratio
        self.res = self.emptyRes() # 初始化每个部门的字典

        self.lanshou = readCX('data/lanshou_.csv')
        self.toudi = readCX('data/toudi_.csv')
        print("读取完成，揽收数据共%d条记录，投递数据共%d条记录" %(len(self.lanshou),len(self.toudi)))
        
        # 合并揽收，投递数据
        self.all = self.lanshou[['投递日期','邮件号','业务种类','分类','重量','投递机构','投递员','投递地址','lng','lat','type','dist','cost','id']]
        self.all =self.all.append(self.toudi[['投递日期','邮件号','业务种类','分类','重量','投递机构','投递员','投递地址','lng','lat','type','dist','cost','id']]).reset_index()
        del self.lanshou
        del self.toudi        
        print("读取完成，揽投数据共%d条记录" % len(self.all))
       
    def genDict(self):
        '''
        生成机构代码到机构简称、经纬度的映射和代价计算结果的映射
        '''
        self.noToName = {} # 机构代码到机构简称
        self.noToLat = {} # 机构代码到机构纬度
        self.noToLng = {} # 机构代码到机构经度
        self.noToRes = {} # 机构代码到结果

        records = readCX(self.department)
        for record in records[['机构代码','机构简称']].values:
            self.noToName[record[0]] = record[1]

        records = readCX(self.select)
        for record in records[['机构代码','lat','lng']].values:
            self.noToLat[record[0]] = record[1]
            self.noToLng[record[0]] = record[2]

    #计算时间代价用到的函数
    def emptyRes(self):
        '''
        初始化res字典，装载所有机构代码
        '''
        res = {}
        for key in self.noToName.keys():
            res[key] = {}
        return res

    def func1(self, x):
        '''
        衰减函数1
        '''
        return np.log2(x + 1) / x

    def func2(self, x):
        '''
        衰减函数2
        '''
        return 1 / (55 + np.log(x))

    def updateDist(self, data, path):
        '''
        以智能体的当前状态的经纬度为依据，计算data中每条记录到其对应机构的距离,存至'dist'列

        参数：
        data -- 需要处理的数据

        返回：
        计算完每条记录至记录对应机构的曼哈顿距离的data
        '''
        departments = readCX(path)
        departments = departments[['机构代码','lat','lng']]

        data['dist'] = ''
        for i in range(len(departments)):
            temp = data.loc[data['投递机构'] == departments.loc[i,:][0], ['lat','lng']]
            dist = latlng2_manhattan_distance(np.array(departments.loc[i,:][1:3]), np.array(temp))
            data.loc[data['投递机构'] == departments.loc[i,:][0], 'dist'] = dist
        
        return data

    def calSemiCost(self, x):
        '''
        计算某投递机构一个投递员在某个上/下午的时间代价

        参数：
        x -- 需要处理的某个上/下午的数据

        返回：
        sum -- 代价
        dist -- 距离
        '''
        if len(x) <= 0: return 0, 0
        
        # 分组，并计算每个组（每个地点所有件）的代价和计数
        group = x[['lng', 'lat', 'cost']].groupby(['lng', 'lat'])
        cost = group.sum()['cost'].values
        count = group.count()['cost'].values.astype(np.float64)
        # 对count进行分段的衰减函数处理
        count = np.piecewise(count, [count < 30, count >= 30], [self.func1, self.func2]) # 分段对件数进行衰减
        # 基础时间+衰减后的时间代价
        sum = (count < 30).sum() / 30 + np.sum(count * cost, axis=0) / self.ratio # 数字太大，除以ratio，近似到每件3min
        
        # 计算通勤时间代价
        x_ = x.drop_duplicates(subset=['lng','lat'],keep='first',inplace=False)
        mean = x_['lat'].mean() # 用均值将记录划分为上下两个部分
        x1 = x_[x_['lat'] >= mean]
        x2 = x_[x_['lat'] < mean]
        if len(x1) < 2 or len(x2) < 2:
            x_ = x_.sort_values(by=['lng','lat'],ascending = [False,False])
            src = np.array(x_[['lng','lat']])[:-1]
            des = np.array(x_[['lng','lat']])[1:]
            dist = (np.sum(latlng2_manhattan_distance(src, des), axis=0)) + x_['dist'].values[0]
        else:
            x1 = x1.sort_values(by=['lng','lat'],ascending = [False,False])
            x2 = x2.sort_values(by=['lng','lat'],ascending = [True,True])
            src = np.array(x1[['lng','lat']])[:-1]
            des = np.array(x1[['lng','lat']])[1:]
            dist = np.sum(latlng2_manhattan_distance(src, des), axis=0) # 上部分距离
            src = np.array(x2[['lng','lat']])[:-1]
            des = np.array(x2[['lng','lat']])[1:]
            dist += np.sum(latlng2_manhattan_distance(src, des), axis=0) # 加上下部分距离
            midist = float(latlng2_manhattan_distance(np.array(x1[['lng','lat']])[-1,:], np.array(x2[['lng','lat']])[0,:]))
            dist += x1['dist'].values[0] + midist + x2['dist'].values[-1] # 加上机构至起始点和上部分至下部分距离

        sum += dist / self.velocity
        
        return sum, dist

    def calCost(self, x):
        '''
        计算某投递机构某投递员在某个上/下午的揽收、投递、总时间代价（和距离）
        '''
        dep_name, date = x[['投递机构','投递员','投递日期']][:1].values[0]
        
        self.res[dep_name][date]['cost'], self.res[dep_name][date]['dist'] = self.calSemiCost(x)

    def calRes(self, path=''):
        '''
        计算代价并将结果保存至csv文件中
        保存在'_man.csv'和'_all.csv'两个文件中

        参数：
        path -- 机构坐标点文件路径
        '''

        # 更新距离信息
        if path != '':
            self.lanshou = self.updateDist(self.lanshou, path)
            self.toudi = self.updateDist(self.toudi, path)
            
        # 计算每个机构的代价
        print("计算每个机构的代价...")
        self.all[['投递机构']].groupby(by=['投递机构']).apply(self.emptyDep) #初始化每个机构的字典
        self.all.grouby(by=['投递机构','投递日期']).apply(self.calCost) #计算代价