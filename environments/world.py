import enum
from algorithms.base import Action
import matplotlib.pyplot as plt
import dm_env
import numpy as np
from abc import abstractmethod
from typing import Any, List, NamedTuple, Tuple
from PIL import Image
from dm_env import specs

from environments import base
from utils.env_info import env_info

Point = Tuple[float, float]

class StandardActions(enum.IntEnum):
    NORTHWEST = 0; NORTH = 1; NORTHEAST = 2
    WEST = 3;      NONE = 4;  EAST = 5
    SOUTHWEST = 6; SOUTH = 7; SOUTHEAST = 8
    
    def vector(self):
        return (
            (-0.1, -0.1), (-0.1, 0.0), (-0.1, 0.1),
            ( 0.0, -0.1), ( 0.0, 0.0), ( 0.0, 0.1),
            ( 0.1, -0.1), ( 0.1, 0.0), ( 0.1, 0.1),
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

class World(base.Environment):
    def __init__(
        self,
        max_steps,
        discount,
        seed: int = None,
        n_action: int = 9
    ):
        super(World, self).__init__()
        global Actions
        if n_action == 4:
            Actions = SmallActions

        # public:
        self.art = None
        self.max_steps = max_steps
        self.discount = discount

        self.bsuite_num_episodes = 10

        # private:
        self._without_draw_item = True
        # TODO(thekips): make this can be used.
        # self._art = np.array([list(x) for x in game_config.art])
        self._rng = np.random.RandomState(seed)
        self._timestep = 0
        
        # self._plot = plt.imshow(np.empty(self.shape))
        
        self._agent_loc = env_info.agent_loc
        self._object_loc = env_info.object_loc

        self._reset()

    def observation_spec(self) -> specs.Array:
        return specs.DiscreteArray(env_info.num_obj(), name="observation")

    def action_spec(self) -> specs.DiscreteArray:
        return specs.DiscreteArray(Actions.num_values(), name="action")

    def _get_observation(self) -> Any:
        # TODO(thekips): make more agent when program can run.
        return (self._agent_loc[52900009], self._object_loc[52900009])
        # return lf._agent_loc

    def _reset(self) -> dm_env.TimeStep:
        # self.art = self._art.copy()
        self._timestep = 0
    
        # reset agent at initial location.
        self._agent_loc = env_info.no_to_location

        return dm_env.restart(self._get_observation())

    def _step(self, action: int) -> dm_env.TimeStep:
        self._timestep += 1

        ## update agent
        for agent in self._agent_loc.keys():
            reward = 0.0
            vector = Actions(action).vector()
            self._agent_loc = (
                max(0, min(self._agent_loc[agent][0] + vector[0], self.shape[0])),
                max(0, min(self._agent_loc[agent][1] + vector[1], self.shape[1])),
            )

        # compute reward by the cost
        cost = self.cost.cal_cost(self._agent_loc)
        reward = sum(cost.values)

        # 增加到最大步数时结束
        if self._timestep == self.max_steps:
            return dm_env.termination(reward, self._get_observation())

        return dm_env.transition(reward, self._get_observation())

    def _draw_item(self,  grid_size: int= 50, gap_size: int= 2):
        blue = (0, 64, 254)
        red = (254, 0, 64)
        yellow = (254, 190, 0)
        green = (0, 199, 159)
        bg_color = (50, 50, 80)

        color_dict = {'#': bg_color, ' ': 'white', 'a': green, 'b': yellow, 'c': red, 'P': blue}

        bg_height = self.art.shape[1] * (grid_size + gap_size) - gap_size
        bg_weight = self.art.shape[0] * (grid_size + gap_size) - gap_size
        bg_size = (bg_height, bg_weight)
        self._img = Image.new('RGB', bg_size, bg_color)

        for i in range(self.art.shape[0]):
            for j in range(self.art.shape[1]):
                if self.art[i][j] != '#':
                    color = color_dict[' ']
                    grid_img = Image.new('RGB', (grid_size, grid_size), color)
                    paste_pos = (j * (grid_size + gap_size), i * (grid_size + gap_size))

                    self._img.paste(grid_img, paste_pos)
        self._grid_img = dict([ (obj.symbol, Image.new('RGB', (grid_size, grid_size), color_dict[obj.symbol])) for obj in self.objects])
        self._grid_img['P'] = Image.new('RGB', (grid_size, grid_size), color_dict['P'])
        
        # 打开动态画图模式
        plt.ion()

        plt.imshow(self._img)
        plt.draw()
        plt.show()
        plt.pause(1)
        plt.clf()
        

    def _plot(self, grid_size: int= 50, gap_size: int= 2):
        img = self._img.copy()
        for obj in self.objects:
            for i, j in self.locate(obj.symbol):
                paste_pos = (j * (grid_size + gap_size), i * (grid_size + gap_size))
                img.paste(self._grid_img[obj.symbol], paste_pos)

        for i, j in self.locate('P'):
            paste_pos = (j * (grid_size + gap_size), i * (grid_size + gap_size))
            img.paste(self._grid_img['P'], paste_pos)


        plt.imshow(img)
        plt.draw()
        plt.show()
        plt.clf()

    def render(self, mode: str = "ansi") -> None:
        if mode == "human":
            if self._without_draw_item:
                self._draw_item()
                self._without_draw_item = False
            self._plot()
        elif mode == "rgb_array":
            return self.art
        elif mode == "ansi":
            print(self.art)
        return

    def bsuite_info(self):
        return {}

