from typing import Dict, Tuple
import numpy as np
import pandas as pd

from utils.get_info import *
from utils.utils import *

class Cost(object):
    def __init__(self) -> None:
        super().__init__()

        env_info = EnvInfo()
        self._velocity = env_info.velocity
        self._ratio = env_info.ratio
        self._data = read_cx('data/data_.csv')
        print("have read %d records" % len(self._data))

    def __func1(self, x) -> np.float64:
        '''
        衰减函数1
        '''
        return np.log2(x + 1) / x

    def __func2(self, x) -> np.float64:
        '''
        衰减函数2
        '''
        return 1 / (55 + np.log(x))

    def __update_dist(self, obj_location: Dict) -> None:
        '''
        Use the new location of agent(department) to update dist column in self._data.
        The dist is distance between record address and its corresponding department.

        Args:
            objects: A dict from department's name to department's instant no and location.

        Returns:
            None
        '''
        self._data['dist'] = self._data.apply(lambda x: location_to_manhattan(obj_location[x.投递机构], tuple(x.lng, x.lat)))

    def __cal_semi_cost(self, x) -> Tuple[np.float64, np.float64]:
        '''
        calculate the cost with input x, and the cost consists of deliver cost and shift cost.

        参数：
        x -- 需要处理的某个上/下午的数据

        返回：
        sum -- 代价
        dist -- 距离
        '''
        if len(x) <= 0:
            return 0, 0

        # group by lng and lat to use discount fuction separately.
        group = x[['lng', 'lat', 'cost']].groupby(['lng', 'lat'])
        cost = group.sum()['cost'].values
        count = group.count()['cost'].values.astype(np.float64)
        # 对count进行分段的衰减函数处理
        count = np.piecewise(count, [count < 30, count >= 30], [self.__func1, self.__func2])  # 分段对件数进行衰减
        # 基础时间+衰减后的时间代价
        sum = (count < 30).sum() / 30 + np.sum(count * cost, axis=0) / self._ratio  # 数字太大，除以ratio，近似到每件3min

        # 计算通勤时间代价
        x_ = x.drop_duplicates(subset=['lng', 'lat'], keep='first', inplace=False)
        mean = x_['lat'].mean()  # 用均值将记录划分为上下两个部分
        x1 = x_[x_['lat'] >= mean]
        x2 = x_[x_['lat'] < mean]
        if len(x1) < 2 or len(x2) < 2:
            x_ = x_.sort_values(by=['lng', 'lat'], ascending=[False, False])
            src = np.array(x_[['lng', 'lat']])[:-1]
            des = np.array(x_[['lng', 'lat']])[1:]
            dist = (np.sum(location_to_manhattan(src, des), axis=0)) + x_['dist'].values[0]
        else:
            x1 = x1.sort_values(by=['lng', 'lat'], ascending=[False, False])
            x2 = x2.sort_values(by=['lng', 'lat'], ascending=[True, True])
            src = np.array(x1[['lng', 'lat']])[:-1]
            des = np.array(x1[['lng', 'lat']])[1:]
            dist = np.sum(location_to_manhattan(src, des), axis=0)  # 上部分距离
            src = np.array(x2[['lng', 'lat']])[:-1]
            des = np.array(x2[['lng', 'lat']])[1:]
            dist += np.sum(location_to_manhattan(src, des), axis=0)  # 加上下部分距离
            midist = float(location_to_manhattan(np.array(x1[['lng', 'lat']])[-1, :], np.array(x2[['lng', 'lat']])[0, :]))
            dist += x1['dist'].values[0] + midist + x2['dist'].values[-1]  # 加上机构至起始点和上部分至下部分距离

        sum += dist / self.velocity

        return tuple(sum, dist) 

    def cal_cost(self, obj_location: Dict) -> Dict:
        '''
        calculate each department's cost and distance.

        Returns:
            A Dict from deparment's name to department's cost.
        '''
        cost = {}
        dist = {}

        def in_cal_cost(x):
            dep_name = x[['投递机构']][:1].values[0]
            cost[dep_name], dist[dep_name] = self.__cal_semi_cost(x)

        # calculate every department's cost separately.
        self.__update_dist(obj_location)
        self._data.grouby(by=['投递机构']).apply(in_cal_cost)

        return cost
