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

    def cal_cost(self, obj_location: Dict) -> Dict:
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
            dep_name = x[['投递机构']][:1].values[0]
            cost[dep_name], dist[dep_name] = self.__cal_semi_cost(x)

        # calculate every department's cost separately.
        self.__update_dist(obj_location)
        self._data.grouby(by=['投递机构']).apply(in_cal_cost)

        return cost