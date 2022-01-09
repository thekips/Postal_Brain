from abc import abstractmethod
import enum
from matplotlib import colors

from numpy import random

import matplotlib.pyplot as plt
import dm_env
import numpy as np
import seaborn as sns
# from abc import abstractmethod

from environments.base import Environment
from utils.env_info import env_info
from utils.heatmap import Image

class StandardActions(enum.IntEnum):
    NORTHWEST = 0; NORTH = 1; NORTHEAST = 2
    WEST = 3;      NONE = 4;  EAST = 5
    SOUTHWEST = 6; SOUTH = 7; SOUTHEAST = 8
    
    def vector(self):
        return (
            (-1, -1), (0, -1), (1, -1),
            (-1,  0), (0,  0), (1,  0),
            (-1,  1), (0,  1), (1,  1),
        )[int(self)]
    @staticmethod
    def num_values():
        return 9

class SmallActions(enum.IntEnum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

    def vector(self):
        return (
            (0, 1), (1, 0), (0, -1), (-1, 0)
        )[int(self)]
    
    @staticmethod
    def num_values():
        return 4

Actions = StandardActions

fig = plt.figure()
grid_size = 10

# rex = lambda x : -1 * np.arctan(x - 6) ** 2
# rey = lambda y : -1 * np.arctan(y - 2) ** 2

# def generate_reward(col, rec):
#     return float(rex(col) + rey(rec))

res = [[0 for i in range(grid_size)] for j in range(grid_size)]
# for i in range(0, grid_size, 1):
#     x = rex(i)
#     for j in range(0, grid_size, 1):
#         y = rey(j)
#         res[j][i] = x + y

for i in range(0, grid_size, 1):
    for j in range(0, grid_size, 1):
        res[i][j] = env_info.opt_value[grid_size - 1 - i][j]
        
def generate_reward(col, rec):
    return env_info.opt_value[grid_size - 1 - rec][col]


class World(Environment):
    def __init__(
        self,
        max_steps,
        tsp_model,
        seed: int = None,
        n_action: int = 9,
    ):
        super(World, self).__init__()
        global Actions
        if n_action == 4:
            Actions = SmallActions

        # public:
        self.art = None
        self.max_steps = max_steps

        self.bsuite_num_episodes = 10

        # private:
        self._without_draw_item = True
        # TODO(thekips): make this can be used.
        # self._art = np.array([list(x) for x in game_config.art])
        self._rng = np.random.RandomState(seed)
        self._timestep = 0
        
        env_info.set_model(tsp_model)

        self._col = grid_size
        self._rec = grid_size

        self._process()
        self._agent_loc = env_info.agent_loc
        self._object_loc = env_info.objects_loc

        # self._old_cost = env_info.cal_cost(self._agent_loc)

        self._reset()

    def _process(self):
        data = env_info._data.drop_duplicates(['lng', 'lat'], keep='first')
        data = data.iloc[:1000]
        print("thekips: len of data is %d." % len(data))
        xx = data['lat'].to_numpy()
        xx = grid_size * (xx - xx.min()) / (xx.max() - xx.min())
        yy = data['lng'].to_numpy()
        yy = grid_size * (yy - yy.min()) / (yy.max() - yy.min())

        env_info.agent_loc = (random.randint(0, grid_size), random.randint(0, grid_size))
        env_info.objects_loc = (xx, yy)

    def observation_spec(self):
        return self.observation.shape

    def action_spec(self):
        return Actions.num_values()

    def _get_observation(self, isplot=False) -> np.ndarray:
        # TODO(thekips): 1.make more agent when program can run. 2.make image to class static.
        # image = Image(self._agent_loc, self._object_loc)
        # print('Gen the kde plot.')
        
        plt.cla()
        # plt.scatter(self._object_loc[0], self._object_loc[1])
        sns.heatmap(res, cmap='Greys', cbar=False)
        # sns.heatmap(res, cmap='Greys', cbar=False, annot=True)
        plt.scatter(x=self._agent_loc[0] + 0.5,y=self._agent_loc[1] + 0.5,c='red')
        plt.axis('off')
        fig.canvas.draw()
        if isplot:
            plt.show(block=False)
            plt.pause(10)
        self.observation = np.frombuffer(fig.canvas.tostring_rgb(),dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # image = Image(self._agent_loc, self._object_loc)
        # self.observation = image.getHeatMap(color)
        # print('Gen the kde plot end.')

        return self.observation

    def _reset(self) -> dm_env.TimeStep:
        # self.art = self._art.copy()
        self._timestep = 0
    
        # reset agent at initial location.
        self._agent_loc = (random.randint(0, grid_size), random.randint(0, grid_size))
        print('reset to loc:', self._agent_loc)
        self._agent_reward = generate_reward(self._agent_loc[0], self._agent_loc[1])

        return self._get_observation()

    @abstractmethod
    def _step(self, action):
        pass

    def _draw_item(self, pos=(7, 6), mode='scatter', is_patch=False, cbar=False, fmt='%.2f'):
        
        plt.cla()
        plt.rcParams['savefig.facecolor']='#efefef'
        # plt.scatter(self._object_loc[0], self._object_loc[1])
        sns.heatmap(res, cmap='Greys', cbar=False)
        # sns.heatmap(res, cmap='Greys', cbar=False, annot=True)
        # if mode == 'scatter':
        #     plt.scatter(self._object_loc[0], self._object_loc[1])
        # elif mode == 'heatmap':
        #     print('draw')
        #     print(res)  #TODEL
        #     sns.heatmap(res, cmap='Greys', cbar=False, annot=True)
        
        plt.scatter(x=pos[0] + 0.5,y=pos[1] + 0.5,c='red')
        if is_patch == True:
            for i in range(grid_size):
                plt.axhline(i, 0, 1, color='w')
                plt.axvline(i, 0, 1, color='w')
        # plt.xlim((0, grid_size))
        # plt.ylim((0, grid_size))
        plt.axis('off')
        fig.canvas.draw()
        plt.show(block=False)

        pname = '_patch' if is_patch else '_nopatch'
        plt.savefig('utils/res/' + mode + pname + '.pdf', bbox_inches='tight', pad_inches=0)
        plt.savefig('utils/res/' + mode + pname + '.jpg', bbox_inches='tight', pad_inches=0)
        plt.pause(9)
        plt.close()

        # image = Image(self._agent_loc, self._object_loc)
        # self.observation = image.getHeatMap(color)
        # print('Gen the kde plot end.')

        return self.observation

    def bsuite_info(self):
        return {}

class ABSWorld(World):
    def _step(self, action):
        action = int(action)
        self._timestep += 1

        # update agent
        vector = Actions(action).vector()
        # print('before is:', self._agent_loc)
        self._agent_loc = (
            min(max(0, self._agent_loc[0] + vector[0]), self._col - 1),
            min(max(0, self._agent_loc[1] + vector[1]), self._rec - 1)
        )
        # print('now is:', self._agent_loc)

        # compute reward by the cost
        reward = generate_reward(self._agent_loc[0], self._agent_loc[1])

        # self._old_cost = cost
        if self._timestep >= self.max_steps:
            return self._reset(), reward

        return self._get_observation(), reward

class RELWorld(World):
    def _step(self, action):
        action = int(action)
        self._timestep += 1

        # update agent
        vector = Actions(action).vector()
        # print('before is:', self._agent_loc)
        self._agent_loc = (
            min(max(0, self._agent_loc[0] + vector[0]), self._col - 1),
            min(max(0, self._agent_loc[1] + vector[1]), self._rec - 1)
        )
        # print('now is:', self._agent_loc)

        # compute reward by the cost
        reward = generate_reward(self._agent_loc[0], self._agent_loc[1]) - self._agent_reward
        self._agent_reward += reward

        # self._old_cost = cost
        if self._timestep >= self.max_steps:
            return self._reset(), reward

        return self._get_observation(), reward