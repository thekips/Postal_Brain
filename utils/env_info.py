import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import yaml

CWD = os.path.dirname(__file__) + '/'

class EnvInfo(object):
    '''
    Generate some info about env information.

    Attributes:
        agent_name: A Dict from department no to department name.
        agent_loc: A Dict from department no to department location.
        velocity: A float representing postman's delivering velocity.
        ratio: A float used to transform weight to hour. 
    '''

    def __init__(self) -> None:
        super().__init__()
        self.agent_name = {} 
        self.agent_loc = {}
        self.object_loc = {}

        # read location of original department's information from config.yaml.
        with open(CWD + 'config.yaml', 'r', encoding='utf-8') as f:
            configs = yaml.safe_load(f)
        self._data = read_cx(CWD + 'data/env.csv')
        print("have read %d records" % len(self._data))

        # some information should share with environment.
        self.__get_agent(configs)
        self.__get_obj()

        self._velocity = configs['velocity']
        self._ratio = configs['ratio']
    
    def __get_agent(self, configs) -> None:
        records = read_cx(CWD + configs['department'])
        for record in records[['机构代码','机构简称','lat','lng']].values:
            self.agent_name[record[0]] = record[1]
            self.agent_loc[record[0]] = (record[2],record[3])
    
    def __get_obj(self) -> None: 
        '''
        use self._data to generate the objects' location.

        Returns:
            object_loc: A dict from agent number to the location of it's objects.
        '''
        def gen_dict(x):
            key = x["投递机构"].unique()[0]
            value = [*zip(x.lat.values, x.lng.values)]
            self.object_loc[key] = value
                
        self._data[['投递机构','lat','lng']].groupby(by=['投递机构']).apply(gen_dict)

    def __discount1(self, x) -> np.float64:
        '''
        when the amount of record in one place less than 30, we use this function to discount.
        '''
        return np.log2(x + 1) / x

    def __discount2(self, x) -> np.float64:
        '''
        when the amount of record in one place more than 30, we use this function to discount.
        '''
        return 1 / (55 + np.log(x))

    def __update_dist(self, agent_loc: Dict) -> None:
        '''
        Use the new location of agent(department) to update dist column in self._data.
        The dist is distance between record address and its corresponding department.

        Args:
            objects: A dict from department's name to department's instant no and location.

        Returns:
            None
        '''
        def func(x: DataFrame):
            aloc = agent_loc[x.iloc[0]['投递机构']]
            self._data.loc[x.index, 'dist']= location_to_manhattan(aloc, x[['lng', 'lat']].values)
        self._data.groupby(by=['投递机构']).apply(func)
        
        # # way 1: by add agent_loc column
        # self._data['dist'] = location_to_manhattan(self._data[['alng', 'alat']], self._data[['lng','lat']]) 
        # print(self._data['dist'])

        # # way 2
        # for k, v in agent_loc:
        #     self._data.loc[self._data['投递机构']==k, 'dist'] = location_to_manhattan(v, self._data.loc[self._data['投递机构']==k, ['lng', 'lat']])
        # print(self._data['dist'])

        # # way 3
        # def func(x):
        #     aloc = agent_loc[x.iloc[0]['投递机构']]
        #     self._data.loc[x.index, 'dist'] = location_to_manhattan(aloc, self._data.loc[x.index, ['lng', 'lat']])
        # self._data[['投递机构','lng', 'lat']].groupby(by=['投递机构']).apply(func)
        # print(self._data['dist'])

        # # way 4
        # def func(x):
        #     aloc = agent_loc[x.iloc[0]['投递机构']]
        #     x.loc['dist'] = location_to_manhattan(aloc, x[['lng', 'lat']])
        # self._data.groupby(by=['投递机构']).apply(func)
        # print(self._data['dist'])

        # # way 5
        # from operator import itemgetter
        # aloc = itemgetter(*self._data['投递机构'].values)(agent_loc)
        # self._data.loc['dist'] = location_to_manhattan(aloc, self._data[['lng'], ['lat']])
        # print(self._data['dist'])


    def __cal_semi_cost(self, x) -> Tuple[np.float64, np.float64]:
        '''
        calculate the sum of cost and dist that a postman should take with input x.
        The sum of cost consists of deliver cost and shift cost.

        Args：
            x: a department's deliver record, which is always in one forenoon or afternoon.

        Returns：
            tuple(sum, dist): a tuple consists of the sum cost and distance.
        '''
        if len(x) <= 0:
            return tuple(0, 0)

        # calculate deliver cost.
        # group by lng and lat to use discount fuction separately.
        group = x[['lng', 'lat', 'cost']].groupby(['lng', 'lat'])
        cost = group.sum()['cost'].values
        count = group.count()['cost'].values.astype(np.float64)
        # use discount function to deal with the record.
        decount = np.piecewise(count, [count < 30, count >= 30], [self.__discount1, self.__discount2])
        # base time + time after discount.
        # divided by ratio to make average cost as 3 min.
        sum = (count < 30).sum() / 30 + np.sum(decount * cost, axis=0) / self._ratio

        # calculate the distance(can be used to calculate shift cost) should take.
        x_ = x.drop_duplicates(subset=['lng', 'lat'], keep='first', inplace=False)
        mean = x_['lat'].mean()  # use mean to divide record as two parts that includes north and south.
        x1 = x_[x_['lat'] >= mean]
        x2 = x_[x_['lat'] < mean]
        if len(x1) < 2 or len(x2) < 2:
            # in this case we do not divide record.
            x_ = x_.sort_values(by=['lng', 'lat'], ascending=[False, False])
            src = np.array(x_[['lng', 'lat']])[:-1]
            des = np.array(x_[['lng', 'lat']])[1:]
            # calculate the sum of distance.
            dist = x_['dist'].value[0] + np.sum(location_to_manhattan(src, des), axis=0) + x_['dist'].values[-1]
        else:
            x1 = x1.sort_values(by=['lng', 'lat'], ascending=[False, False])
            x2 = x2.sort_values(by=['lng', 'lat'], ascending=[True, True])
            # calculate north distance.
            src = np.array(x1[['lng', 'lat']])[:-1]
            des = np.array(x1[['lng', 'lat']])[1:]
            dist = np.sum(location_to_manhattan(src, des), axis=0)
            # calculate south distance.
            src = np.array(x2[['lng', 'lat']])[:-1]
            des = np.array(x2[['lng', 'lat']])[1:]
            dist += np.sum(location_to_manhattan(src, des), axis=0)
            # calculate the distance between last point in north and first point in south.
            midist = float(location_to_manhattan(np.array(x1[['lng', 'lat']])[-1, :], np.array(x2[['lng', 'lat']])[0, :]))
            # calculate the sum of distance.
            dist += x1['dist'].values[0] + midist + x2['dist'].values[-1] 

        sum += dist / self._velocity # add deliver cost.
        return (sum, dist) 

    def cal_cost(self, agent_loc: Dict) -> Dict:
        '''
        calculate each department's instant cost and distance.

        Args:
            obj_location: A dict from department's no to department's location(a tuple include lng and lat).

        Returns:
            A Dict from deparment's name to department's instant cost.
        '''
        cost = {}
        dist = {}

        def in_cal_cost(x):
            dep_name = x.iloc[0]['投递机构']
            # TODO(thekips): add other dep_name to cal cost.
            if dep_name != 52900009 and dep_name != '52900009':
                cost[dep_name], dist[dep_name] = 0, 0
                return
            cost[dep_name], dist[dep_name] = self.__cal_semi_cost(x)

        # calculate every department's cost separately.
        self.__update_dist(agent_loc)
        self._data.groupby(by=['投递机构']).apply(in_cal_cost)

        return cost

    def num_obj(self) -> int:
        return len(self._data)

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
    d_lat_lon = np.abs(np.radians(loc1) - np.radians(loc2))
    
    r = 6373.0
    a_lat_lon = np.sin(d_lat_lon / 2.0) **2
    distance = 2 * np.arctan2(np.sqrt(a_lat_lon), np.sqrt(1 - a_lat_lon))
    distance = r * distance
    distance = distance.reshape(-1,2)
    distance = np.sum(distance, axis=1)
    
    return distance if len(distance) > 1 else distance[0]

env_info = EnvInfo()
