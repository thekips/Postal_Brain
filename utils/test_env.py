from datetime import time
import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import yaml

from utils.typedef import Point

CWD = os.path.dirname(__file__) + '/'

def read_csv(path, low_memory=False) -> DataFrame:
    '''
    read a csv file with encoding=gb18030.

    Args：
        path: the path of csv file.
        low_memory: whether read file in low_memory mode.

    Returns:
        A dataframe object which include the content of file with the given path.
    '''
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
        self.agent_name: dict
        self.agent_loc: dict
        self.object_loc: dict

        # read location of original department's information from config.yaml.
        with open(CWD + 'config.yaml', 'r', encoding='utf-8') as f:
            configs = yaml.safe_load(f)
        self._data = read_cx(CWD + 'data/data_.csv')
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

    def _update_dist(self, agent_loc: Dict) -> None:
        '''
        Use the new location of agent(department) to update dist column in self._data.
        The dist is distance between record address and its corresponding department.

        Args:
            objects: A dict from department's name to department's instant no and location.

        Returns:
            None
        '''
        
        func = lambda x: location_to_manhattan(agent_loc[x.投递机构], (x.lng, x.lat))
        self._data['dist'] = self._data[['投递机构','lng','lat']].apply(func, axis=1)

        # TODO(thekips):try to change code below to run fast . And we can try cupy and only save [['投递机构','lng','lat','cost']].

        # # way 1: by add agent_loc column
        # self._data['dist'] = location_to_manhattan(self._data[['alng', 'alat']], self._data[['lng','lat']]) 
        # print(self._data['dist'])

        # # way 2
        # for k, v in agent_loc:
        #     self._data.loc[self._data['投递机构']==k, 'dist'] = location_to_manhattan(v, self._data.loc[self._data['投递机构']==k, ['lng', 'lat']])
        # print(self._data['dist'])

        # #way 3
        # def func(x):
        #     aloc = agent_loc[x.iloc[0]['投递机构']]
        #     self._data.loc[x.index, 'dist'] = location_to_manhattan(aloc, self._data.loc[x.index, ['lng', 'lat']])
        # self._data[['投递机构','lng', 'lat']].groupby(by=['投递机构']).apply(func)
        # print(self._data['dist'])

        # #way 4
        # def func(x):
        #     aloc = agent_loc[x.iloc[0]['投递机构']]
        #     x.loc['dist'] = location_to_manhattan(aloc, x[['lng', 'lat']])
        # self._data.groupby(by=['投递机构']).apply(func)
        # print(self._data['dist'])

        # #way 5
        # from operator import itemgetter
        # aloc = itemgetter(*self._data['投递机构'].values)(agent_loc)
        # self._data.loc['dist'] = location_to_manhattan(aloc, self._data[['lng'], ['lat']])
        # print(self._data['dist'])



    def test(self, agent_loc: Dict) -> Dict:
        '''
        calculate each department's instant cost and distance.

        Args:
            obj_location: A dict from department's no to department's location(a tuple include lng and lat).

        Returns:
            A Dict from deparment's name to department's instant cost.
        '''
        # calculate every department's cost separately.
        print("Update dist.")
        btime = time.time()
        self.__update_dist(agent_loc)
        etime = time.time()
        print("Finish update dist.")
        print("Totally %ds", etime - btime)


env_info = EnvInfo()