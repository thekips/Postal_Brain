import os
from time import time
import torch
from typing import Dict, List
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import yaml
import json
import scipy.stats as stats
CWD = os.path.dirname(__file__) + '/'

def zscore(x: Dict):
    x = sorted(x.items(), key=lambda z : z[0])
    x = [-z[1] for z in x]
    x = stats.zscore(x)
    dim = int(pow(len(x), 0.5))
    res = [[0 for i in range(dim)] for j in range(dim)]
    for i in range(dim):
        for j in range(dim):
            res[i][j] = float(x[dim * i + j])
    
    return res

# def calDeliverCost(x: DataFrame, ratio: float):
#     '''
#     Args: 
#         x: a dataframe should have columns [['lat', 'lng', 'cost']].
#         ratio: an float to make average cost as 3 min.
#     '''
#     group = x[['lng', 'lat', 'cost']].groupby(['lng', 'lat'])
#     cost = group.sum()['cost'].values
#     count = group.count()['cost'].values.astype(np.float64)

#     # base time + time after discount.
#     base_time = (count < 30).sum() / 30
#     # use  discount to deal with the record.
#     discount1 = lambda x : np.log2(x + 1) / x
#     discount2 = lambda x : 1 / (55 + np.log(x))
#     decount = np.piecewise(count, [count < 30, count >= 30], [discount1, discount2])
#     discount_time = np.sum(decount * cost, axis=0) / ratio

#     return base_time + discount_time

def read_csv(path, usecols, low_memory=False) -> DataFrame:
    '''
    read a csv file with encoding=gb18030.

    Args：
        path: the path of csv file.
        low_memory: whether read file in low_memory mode.

    Returns:
        A dataframe object which include the content of file with the given path.
    '''
    try:
        return pd.read_csv(path, encoding='gb18030', usecols=usecols, low_memory=low_memory)
    except:
        return pd.read_csv(path, usecols=usecols, low_memory=low_memory)

def read_cx(path, usecols=None) -> DataFrame:
    '''
    Read a csv file or excel file like '.xlsx', '.xls'.

    Args:
        path: the path of csv file or excel file.

    Returns:
        A dataframe object which include the content of file with the given path.
    '''
    try:
        return read_csv(path, usecols)
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
    abs = np.abs(np.array(loc1) - np.array(loc2))
    return np.sum(abs)

class EnvInfo(object):
    '''
    Generate some info about env information.

    Attributes:
        agent_name: A Dict from department no to department name.
        agent_loc: A Dict from department no to department location.
        velocity: A float representing postman's delivering velocity.
        ratio: A float used to transform weight to hour. 
    '''

    def __init__(self, data_path, dist_path) -> None:
        super().__init__()
        self.agent_name = {} 
        self.agent_loc = {}

        # read location of original department's information from config.yaml.
        with open(CWD + 'config.yaml', 'r', encoding='utf-8') as f:
            configs = yaml.safe_load(f)
        self._data = read_cx(CWD + data_path)
        self.__get_agent(configs)
        self.__process()
        with open(CWD + 'dist/true_opt_value.json') as f:
            self.reward = json.load(f)
        self.reward = zscore(self.reward)
        

        # some information should share with environment.
        # self.__dist_path = CWD + dist_path
        # self.__get_dist(self.__dist_path)

        # self._velocity = configs['velocity']
        # self._ratio = configs['ratio']

    def __get_agent(self, configs) -> None:
        records = read_cx(CWD + configs['department'])
        for record in records[['机构代码','机构简称','lat','lng']].values:
            self.agent_name[record[0]] = record[1]
            self.agent_loc[record[0]] = (record[2],record[3])
    
    def __process(self):
        print("have read %d records" % len(self._data))
        x = self._data.iloc[:1000].groupby(['lng','lat']).sum()
        print("after delete dulplicate, %d records" % len(x))

        location = x.index.to_numpy()
        self.objects_wei = x['重量'].to_numpy()
        assert(self.objects_wei.shape[0] == location.shape[0])

        lng = np.array([x[0] for x in location])
        lat = np.array([x[1] for x in location])
        lng_min = min(lng)
        lng_abs = max(lng) - lng_min
        lat_min = min(lat)
        lat_abs = max(lat) - lat_min
        lat = (lat - lat_min) / lat_abs
        lng = (lng - lng_min) / lng_abs
        self.objects_loc = [*zip(lat, lng)]

        self.agent_loc = (lat.mean(), lng.mean())
        lat = (self.agent_loc[0] - lat_min) / lat_abs
        lng = (self.agent_loc[1] - lng_min) / lng_abs 
        self.agent_loc = (lat, lng) #TODO(thekips): Multi-Agent
    
    # def __get_dist(self, path):
    #     if os.path.exists(path):
    #         with open(path, 'r') as f:
    #             self.__distance = json.load(f)
    #     else:
    #         self.__distance = {}
        
    def __cal_one(self, agent_loc: tuple):
        '''
        Args:
            x: a dataframe should have columns [['lat', 'lng']], start point should be at 0 line.
            velocity: a float(km/h) to calculate time of shift, to turn dist(km) to time(h).
        '''
        time1 = time()
        data = [agent_loc] + self.objects_loc
        data = torch.Tensor(data).cuda()
        mask = torch.zeros(data.shape[0]).cuda()
        solution = []   #recording the coordination.
        burden = 0  #recording the weight cost.

        mask = mask.unsqueeze(0)
        data = data.unsqueeze(0)
        x = data[:, 0, :]
        h = None
        c = None
        time2 = time()
        for i in range(data.shape[1]):
            out, h, c, _ = self._model(x=x, X_all=data, h=h, c=c, mask=mask)
            idx = torch.argmax(out, dim=1)

            burden += (i + 1) * self.objects_wei[idx - 1] 
            x = data[0, idx]
            solution.append(x.cpu().numpy())
            mask[0, idx] += -np.inf

        time3 = time()
        solution.append(solution[0])
        solution = np.array(solution)
        solution = solution.squeeze()
        distance = location_to_manhattan(solution[:-1], solution[1:])
        time4 = time()

        print(time2 - time1, " ", time3 - time2, " ", time4 - time3)
        # # Plot image.
        # x_coor,y_coor = zip(*solution)
        # plt.step(x_coor, y_coor, label='manhattan')
        # plt.show()

        return distance + burden

    def set_model(self, tsp_model):
        self._model = tsp_model

    def cal_cost(self, agent_loc: Dict) -> Dict:
        '''
        calculate each department's instant cost and distance.

        Args:
            obj_location: A dict from department's no to department's location(a tuple include lng and lat).

        Returns:
            A Dict from deparment's name to department's instant cost.
        '''

        cost = self.__cal_one(agent_loc)

        # self.__distance[str(agents_loc)] = cost
        # with open(self.__dist_path, 'w') as f:
        #     json.dump(self.__distance, f, ensure_ascii=False, indent=4)

        return cost

    def num_obj(self) -> int:
        return len(self._data)

env_info = EnvInfo('data/env.csv', 'dist/test.csv')
