import csv
import numpy as np
import pandas as pd
import time

from utils import makeDir, readCX, latlng2_manhattan_distance

class Cost:
    def __init__(self, velocity, ratio, noToName, noToLat, noToLng, id2Name):
        self.velocity = velocity
        self.ratio = ratio
        self.noToName = noToName
        self.noToLat = noToLat
        self.noToLng = noToLng
        self.id2Name = id2Name

        self.res_path = 'res/'

        self.lanshou = readCX('data/lanshou_.csv')
        self.toudi = readCX('data/toudi_.csv')
        print("读取完成，揽收数据共%d条记录，投递数据共%d条记录" %(len(self.lanshou),len(self.toudi)))
       
    #计算时间代价用到的函数
    def emptyRes(self):
        '''
        初始化res字典，装载所有机构代码
        '''
        res = {}
        for key in self.noToName.keys():
            res[key] = {}
        return res

    def emptyDep(self, x):
        '''
        向res字典装载所有投递员的字典
        '''
        self.res[x['投递机构'].values[0]][x['投递员'].values[0]] = {}

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
        以path中的机构经纬度为依据，计算data中每条记录到记录对应机构的曼哈顿距离,存至'dist'列

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
        计算某投递机构某投递员在某个上/下午的时间代价
        '''
        if len(x) <= 0: return 0, 0
        
        # 分组，并计算每个组（每个地点所有件）的代价和计数
        group = x[['lng', 'lat', 'cost']].groupby(['lng', 'lat'])
        cost = group.sum()['cost'].values
        count = group.count()['cost'].values.astype(np.float64)
        # 对count进行分段的衰减函数处理
        count = np.piecewise(count, [count < 30, count >= 30], [self.func1, self.func2]) # 分段对件数进行衰减
        # 基础时间+衰减后的时间代价
        sum1 = (count < 30).sum() / 30 + np.sum(count * cost, axis=0) / self.ratio # 数字太大，除以ratio，近似到每件3min
        
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

        sum3 = dist / self.velocity
        
        return sum1 + sum3, dist

    def calCost(self, x):
        '''
        计算某投递机构某投递员在某个上/下午的揽收、投递、总时间代价（和距离）
        '''
        dep_name, road, date = x[['投递机构','投递员','投递日期']][:1].values[0]
        x_t = x[~x['id']]
        x_l = x[x['id']]
        self.res[dep_name][road][date] = {}
        self.res[dep_name][road][date]['t0'] = len(x_t)
        self.res[dep_name][road][date]['t1'] = len(x_t[x_t['分类']==1.2])
        self.res[dep_name][road][date]['t2'] = len(x_t[x_t['分类']==1.5])
        self.res[dep_name][road][date]['t3'] = len(x_t[x_t['分类']==2])
        self.res[dep_name][road][date]['t4'] = len(x_t[x_t['分类']==4])
        self.res[dep_name][road][date]['l0'] = len(x_l)
        self.res[dep_name][road][date]['l1'] = len(x_l[x_l['分类']==1.2])
        self.res[dep_name][road][date]['l2'] = len(x_l[x_l['分类']==1.5])
        self.res[dep_name][road][date]['l3'] = len(x_l[x_l['分类']==2])
        self.res[dep_name][road][date]['l4'] = len(x_l[x_l['分类']==4])
        
        self.res[dep_name][road][date]['tcost'], self.res[dep_name][road][date]['tdist'] = self.calSemiCost(x_t)
        self.res[dep_name][road][date]['lcost'], self.res[dep_name][road][date]['ldist'] = self.calSemiCost(x_l)
        self.res[dep_name][road][date]['cost'], self.res[dep_name][road][date]['dist'] = self.calSemiCost(x)

    def calRes(self, path=''):
        '''
        计算代价并将结果保存至csv文件中
        保存在'_man.csv'和'_all.csv'两个文件中

        参数：
        path -- 机构坐标点文件路径
        '''
        noToName = self.noToName
        noToLat = self.noToLat
        noToLng = self.noToLng
        id2Name = self.id2Name

        # 更新距离信息
        if path != '':
            self.lanshou = updateDist(self.lanshou, path)
            self.toudi = updateDist(self.toudi, path)
            
        # 保存揽收投递总数信息
        infos = {}
        for key in noToName.keys():
            infos[key] = {}
            infos[key]['t0'] = len(self.toudi[self.toudi['投递机构']==key])
            infos[key]['t1'] = len(self.toudi[(self.toudi['投递机构']==key) & (self.toudi['分类']==1.2)])
            infos[key]['t2'] = len(self.toudi[(self.toudi['投递机构']==key) & (self.toudi['分类']==1.5)])
            infos[key]['t3'] = len(self.toudi[(self.toudi['投递机构']==key) & (self.toudi['分类']==2)])
            infos[key]['t4'] = len(self.toudi[(self.toudi['投递机构']==key) & (self.toudi['分类']==4)])
            infos[key]['l0'] = len(self.lanshou[self.lanshou['投递机构']==key])
            infos[key]['l1'] = len(self.lanshou[(self.lanshou['投递机构']==key) & (self.lanshou['分类']==1.2)])
            infos[key]['l2'] = len(self.lanshou[(self.lanshou['投递机构']==key) & (self.lanshou['分类']==1.5)])
            infos[key]['l3'] = len(self.lanshou[(self.lanshou['投递机构']==key) & (self.lanshou['分类']==2)])
            infos[key]['l4'] = len(self.lanshou[(self.lanshou['投递机构']==key) & (self.lanshou['分类']==4)])
            
        # 合并揽收，投递数据
        data_alls = self.lanshou[['投递日期','邮件号','业务种类','分类','重量','投递机构','投递员','投递地址','lng','lat','type','dist','cost','id']]
        data_alls = data_alls.append(self.toudi[['投递日期','邮件号','业务种类','分类','重量','投递机构','投递员','投递地址','lng','lat','type','dist','cost','id']]).reset_index()
        del self.lanshou
        del self.toudi
        
        # 计算每个人的代价
        print("计算每个人的代价...")
        self.res = self.emptyRes() # 初始化每个部门的字典
        data_alls[['投递机构','投递员']].groupby(by=['投递机构','投递员']).apply(self.emptyDep) #初始化每个投递员的字典
        data_alls.groupby(by=['投递机构','投递员','投递日期']).apply(self.calCost) #计算代价
        
        # 计算各部门工作人次信息
        days = {}
        temp_alls = data_alls[['投递机构','投递员']].copy(deep=True)
        temp_alls['投递日期'] = data_alls['投递日期'].apply(lambda x: x[:-3])
        for key in self.res.keys():
            days[key] = len(temp_alls[temp_alls['投递机构']==key].groupby(by=['投递机构','投递员','投递日期']))
        del temp_alls
        
        print("写入结果...")
        #处理文件保存路径
        makeDir(self.res_path)
        prefix = input("请输入保存文件的前缀，留空则默认为res：")
        if len(prefix) == 0: prefix = 'res'
        # 写入每个人每个时间段的信息
        header = ['揽投员','机构代码','机构简称','日期',r'投递总数(件)',r'快包(件)',r'标快(件)',r'增值(件)',r'高端(件)',
                r'揽收总数(件)',r'快包(件)',r'标快(件)',r'增值(件)',r'高端(件)','投递代价','投递距离','揽收代价','揽收距离','总代价','总距离']

        costs = {} # 总时间代价
        tcosts = {} # 投递时间代价
        lcosts = {} # 揽收时间代价
        dists = {} # 总距离
        with open(self.res_path + prefix + '_man.csv','w',encoding='gb18030',newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            for key1 in self.res.keys():
                costs[key1] = 0
                tcosts[key1] = 0
                lcosts[key1] = 0
                dists[key1] = 0
                for key2 in self.res[key1].keys():
                    for key3 in self.res[key1][key2].keys():
                        temp = self.res[key1][key2][key3]
                        costs[key1] += temp['cost']
                        tcosts[key1] += temp['tcost']
                        lcosts[key1] += temp['lcost']
                        dists[key1] += temp['dist']
                        w.writerow([id2Name.get(key2), key1, noToName[key1], key3, temp['t0'], temp['t1'], temp['t2'],
                                        temp['t3'], temp['t4'], temp['l0'], temp['l1'], temp['l2'], temp['l3'], temp['l4'],
                                        temp['tcost'], temp['tdist'], temp['lcost'], temp['ldist'], temp['cost'], temp['dist']])
                        
        # 写入每个机构的信息
        header = ['机构','机构简称','经度','纬度','总代价','总工作人次','人均日代价','投递代价','揽收代价',r'投递总数(件)',
                r'快包(件)',r'标快(件)',r'增值(件)',r'高端(件)',r'揽收总数(件)',r'快包(件)',r'标快(件)',r'增值(件)',r'高端(件)']

        with open(self.res_path + prefix + '_all.csv','w',encoding='gb18030',newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            for key in self.res.keys():
                w.writerow([key, noToName[key], noToLng[key], noToLat[key], costs[key], days[key], costs[key]/days[key],
                            tcosts[key], lcosts[key], infos[key]['t0'], infos[key]['t1'], infos[key]['t2'], infos[key]['t3'], 
                            infos[key]['t4'], infos[key]['l0'], infos[key]['l1'], infos[key]['l2'], infos[key]['l3'], infos[key]['t4']])
        print("计算完毕，结果保存在res文件夹下")