import enum

from numpy import random

import matplotlib.pyplot as plt
import dm_env
import numpy as np
import seaborn as sns

from environments.base import Environment
from utils.env_info import env_info
from utils.heatmap import Image

class StandardActions(enum.IntEnum):
    NORTHWEST = 0; NORTH = 1; NORTHEAST = 2
    WEST = 3;      NONE = 4;  EAST = 5
    SOUTHWEST = 6; SOUTH = 7; SOUTHEAST = 8
    
    def vector(self):
        return (
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1), ( 0, 0), ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
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

# def generate_reward(col, rec):
#     x = -1 * (col / 10 - 4) ** 2 + 8
#     y = -1 * (rec / 10 - 6) ** 2 + 12
#     return x + y

# res = [[0 for i in range(100)] for j in range(100)]
# for i in range(0, 100, 1):
#     x = -1 * (i / 10 - 4) ** 2 + 8
#     for j in range(0, 100, 1):
#         y = -1 * (j / 10 - 6) ** 2 + 12
#         res[i][j] = x + y
fig = plt.figure()
grid_size = 10

rex = lambda x : np.arctan(x - 7) ** 2 + np.cos(2 * x)
rey = lambda y : np.arctan(y - 7) ** 2 + np.cos(2 * y)

def generate_reward(col, rec):
    return float(rex(col) + rey(rec))

res = [[0 for i in range(grid_size)] for j in range(grid_size)]
for i in range(0, grid_size, 1):
    x = rex(i)
    for j in range(0, grid_size, 1):
        y = rey(j)
        res[j][i] = x + y

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
        
        # self._plot = plt.imshow(np.empty(self.shape))
        
        env_info.set_model(tsp_model)

        env_info.agent_loc = (random.randint(0, grid_size), random.randint(0, grid_size))
        env_info.objects_loc = res
        self._col = grid_size
        self._rec = grid_size

        self._agent_loc = env_info.agent_loc
        self._object_loc = env_info.objects_loc

        # self._old_cost = env_info.cal_cost(self._agent_loc)

        print('Environment reset.')
        self._reset()

    def observation_spec(self):
        return self.observation.shape

    def action_spec(self):
        return Actions.num_values()

    def _get_observation(self, isplot=False) -> np.ndarray:
        # TODO(thekips): 1.make more agent when program can run. 2.make image to class static.
        # image = Image(self._agent_loc, self._object_loc)
        # print('Gen the kde plot.')
        
        plt.cla()
        sns.heatmap(res, cmap='Greys', cbar=False)
        plt.scatter(x=self._agent_loc[0] + 0.5,y=self._agent_loc[1] + 0.5,c='red')
        plt.axis('off')
        fig.canvas.draw()
        if isplot:
            plt.show(block=False)
            plt.pause(0.05)
        self.observation = np.frombuffer(fig.canvas.tostring_rgb(),dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # image = Image(self._agent_loc, self._object_loc)
        # self.observation = image.getHeatMap(color)
        # print('Gen the kde plot end.')

        return self.observation

    def _reset(self) -> dm_env.TimeStep:
        # self.art = self._art.copy()
        print('thekips: reset env.')
        self._timestep = 0
    
        # reset agent at initial location.
        self._agent_loc = (random.randint(0, grid_size), random.randint(0, grid_size))
        print('reset to loc:', self._agent_loc)
        self._agent_reward = generate_reward(self._agent_loc[0], self._agent_loc[1])

        return self._get_observation()

    def _step(self, action):
        action = int(action)
        self._timestep += 1

        # update agent
        vector = Actions(action).vector()
        self._agent_loc = (
            min(max(0, self._agent_loc[0] + vector[0]), self._col - 1),
            min(max(0, self._agent_loc[1] + vector[1]), self._rec - 1)
        )

        # compute reward by the cost
        reward = generate_reward(self._agent_loc[0], self._agent_loc[1]) - self._agent_reward
        self._agent_reward += reward

        # self._old_cost = cost
        if self._timestep >= self.max_steps:
            return self._reset(), reward

        return self._get_observation(), reward

    def _draw_item(self):
        plt.clf()
        plt.imshow(self.observation)
        plt.show()

    def bsuite_info(self):
        return {}

