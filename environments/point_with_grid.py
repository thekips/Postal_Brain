import enum
from utils.cost import Cost
import matplotlib.pyplot as plt
import dm_env
import numpy as np
from abc import abstractmethod
from typing import Any, List, NamedTuple, Tuple
from PIL import Image
from dm_env import specs

from environments import base

Point = Tuple[int, int]

class GridworldObject(NamedTuple):
    location: Tuple(np.double, np.double) 

class GridworldConfig(NamedTuple):
    art: List[str]
    objects: Tuple[GridworldObject]
    max_steps: int
    discount: float = 0.99

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

class Gridworld(base.Environment):
    def __init__(
        self,
        game_config: GridworldConfig,
        seed: int = None,
        n_action: int = 9
    ):
        super(Gridworld, self).__init__()
        if n_action == 4:
            Actions = SmallActions

        # public:
        self.art = None
        self.objects = game_config.objects
        self.max_steps = game_config.max_steps
        self.discount = game_config.discount
        self.shape = (len(game_config.art), len(game_config.art[0]))

        self.bsuite_num_episodes = 10

        # private:
        self._without_draw_item = True
        self._art = np.array([list(x) for x in game_config.art])
        self._rng = np.random.RandomState(seed)
        self._timestep = 0
        
        # self._plot = plt.imshow(np.empty(self.shape))
        
        self._agent_location = None
        self._object_locations = dict()

        self._reset()

    @abstractmethod
    def _get_observation(self) -> Any:
        raise NotImplementedError

    def _reset(self) -> dm_env.TimeStep:
        self.art = self._art.copy()
        self._timestep = 0
    
        # spawn agent at random location
        self._agent_location = self.spawn('P')

        return dm_env.restart(self._get_observation())

    def _step(self, action: int) -> dm_env.TimeStep:
        self._timestep += 1

        ## update agent
        for _object in self.objects:
            reward = 0.0
            vector = Actions(action).vector()
            location = (
                max(0, min(self._agent_location[0] + vector[0], self.shape[0])),
                max(0, min(self._agent_location[1] + vector[1], self.shape[1])),
            )

            # set new agent position
            _object.location = location

        # compute reward by the cost
        reward = 

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

    @abstractmethod
    def observation_spec(self) -> specs.Array:
        raise NotImplementedError

    def action_spec(self) -> specs.DiscreteArray:
        return specs.DiscreteArray(Actions.num_values(), name="action")

    def random_point(self) -> Point:
        return (
            self._rng.randint(0, self.shape[0]),
            self._rng.randint(0, self.shape[1]),
        )

    def locate(self, symbol: chr) -> List[Point]:
        i, j = np.where(self.art == symbol)
        locations = list(zip(i, j))
        return locations

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


class TabularGridworld(Gridworld):
    # def __init__(self, game_config: GridworldConfig, seed: int = None):
    #     super().__init__(game_config, seed=seed)    

    def _get_observation(self) -> Any:
        i, j = self.locate("P")[0]
        return i * self.shape[0] + j

    def observation_spec(self) -> specs.DiscreteArray:
        #print('tabular shape = ', self.shape, sum(self.shape))
        num_values = self.shape[0] * self.shape[1]
        return specs.DiscreteArray(num_values=num_values, name="observation")

    def _fix_object_locations(self) -> None:
        self._object_locations = {obj.symbol: [] for obj in self.objects}
        for obj in self.objects:
            for _ in range(obj.N):
                location = self.empty_point()
                self._object_locations[obj.symbol].append(location)
                self.art[location] = obj.symbol
        return

VERY_DENSE = GridworldConfig(
    art=[
        "#############"
        "#           #"
        "#           #"
        "#           #"
        "#           #"
        "#           #"
        "#           #"
        "#           #"
        "#           #"
        "#           #"
        "#           #"
        "#           #"
        "#############"
    ],
    objects=tuple(
        map(
            lambda x: GridworldObject(*x),
            [
                (1, 1.0, 0.0, 1.0, " "),
            ],
        )
    ),
    max_steps=2000,
)

class GridMaps(NamedTuple):
    VERY_DENSE = VERY_DENSE