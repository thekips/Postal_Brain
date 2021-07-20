import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import yaml

from utils.typedef import Point

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
        # read location of original department's information from config.yaml.
        with open(CWD + 'config.yaml', 'r', encoding='utf-8') as f:
            configs = yaml.safe_load(f)

        self._velocity = configs['velocity']
        self._ratio = configs['ratio']

        self._data = read_cx(CWD + 'data/data_.csv')
        print("have read %d records" % len(self._data))

        # some information should share with environment.
        self.agent_name = {}
        self.agent_loc = {}

        records = read_cx(CWD + configs['department'])
        for record in records[['机构代码','机构简称','lat','lng']].values:
            self.agent_name[record[0]] = record[1]
            self.agent_loc[record[0]] = (record[2],record[3])
        
        self.object_loc = self._get_obj()
    
    def _get_obj(self) -> Dict[str, List[Point]]: 
        '''
        use self._data to generate the objects' location.

        Returns:
            object_loc: A dict from agent number to the location of it's objects.
        '''
        object_loc = {}

        def gen_dict(x):
            key = x.投递机构.unique()[0]
            value = [*zip(x.lat.values, x.lng.values)]
            object_loc[key] = value
                
        self._data[['投递机构','lat','lng']].groupby(by=['投递机构']).apply(gen_dict)

        return object_loc

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
        self._data['dist'] = self._data.apply(lambda x: location_to_manhattan(agent_loc[x.投递机构], tuple(x.lng, x.lat)))

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
        count = np.piecewise(count, [count < 30, count >= 30], [self.__discount1, self.__discount2])
        # base time + time after discount.
        # divided by ratio to make average cost as 3 min.
        sum = (count < 30).sum() / 30 + np.sum(count * cost, axis=0) / self._ratio

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

        sum += dist / self.velocity # add deliver cost.
        return tuple(sum, dist) 

    def cal_cost(self, agent_loc: Dict) -> Dict:
        '''
        calculate each department's instant cost and distance.

        Args:
            obj_location: A dict from department's no to department's location(a tuple include lng and lat).

        Returns:
            A Dict from deparment's name to department's instant cost.
        '''
        cost: dict
        dist: dict

        def in_cal_cost(x):
            dep_name = x[['投递机构']][:1].values[0]
            cost[dep_name], dist[dep_name] = self.__cal_semi_cost(x)

        # calculate every department's cost separately.
        self.__update_dist(agent_loc)
        self._data.grouby(by=['投递机构']).apply(in_cal_cost)

        return cost

    def num_obj(self) -> int:
        return len(self._data)

def read_csv(path, low_memory=False) -> DataFrame:
    '''
    read a csv file with encoding=gb18030.

    Args：
        path: the path of csv file.
        low_memory: whether read file in low_memory mode.

    Returns:
        A dataframe object which include the content of file with the given path.
    '''
    print(path)
    try:
        return pd.read_csv(path,encoding='gb18030',low_memory=low_memory)
    except:
        return pd.read_csv(path,low_memory=low_memory)

def read_cx(path) -> DataFrame:
    '''
    Read a csv file or excel file like '.xlsx', '.xls'.

    Args:
        path: the path of csv file or excel file.

    Returns:
        A dataframe object which include the content of file with the given path.
    '''
    try:
        return read_csv(path)
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