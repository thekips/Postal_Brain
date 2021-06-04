import enum
from abc import abstractmethod
from typing import Any, List, NamedTuple, Tuple

import matplotlib.pyplot as plt
from PIL import Image
from lrla.environments import base
import dm_env
import numpy as np
from dm_env import specs


Point = Tuple[int, int]

class GridworldObject(NamedTuple):
    N: int
    reward: float
    eps_term: float
    eps_respawn: float
    symbol: chr


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
        for obj in self.objects:
            self._object_locations[obj.symbol] = [None] * obj.N

        self._reset()

    @abstractmethod
    def _get_observation(self) -> Any:
        raise NotImplementedError

    def _reset(self) -> dm_env.TimeStep:
        self.art = self._art.copy()
        self._timestep = 0
    
        # spawn objects at random location
        for obj in self.objects:
            for i in range(obj.N):
                # print(self._object_locations[obj.symbol][i])
                self._object_locations[obj.symbol][i] = self.spawn(obj.symbol, 
                                        self._object_locations[obj.symbol][i])

        # spawn agent at random location
        self._agent_location = self.spawn('P')

        return dm_env.restart(self._get_observation())

    def _step(self, action: int) -> dm_env.TimeStep:
        self._timestep += 1

        ## update agent
        reward = 0.0
        vector = Actions(action).vector()
        location = (
            max(0, min(self._agent_location[0] + vector[0], self.shape[0])),
            max(0, min(self._agent_location[1] + vector[1], self.shape[1])),
        )
        # hit a wall, go back (diagonal moves are never done partially)
        if self.art[location] == "#":
            location = self._agent_location
    
        # stepped on object, compute reward
        if self.art[location] in [obj.symbol for obj in self.objects]:
            obj = [x for x in self.objects if x.symbol == self.art[location]]
            if len(obj) > 0:
                reward = obj[0].reward
                #  termination probability
                if self._rng.random() < obj[0].eps_term:
                    return dm_env.termination(reward, self._get_observation())
            
        # set new agent position
        self.art[self._agent_location] = " "
        self.art[location] = "P"
        self._agent_location = location

        ## update environment, let it be ❤
        for obj in self.objects:
            for i, location in enumerate(self._object_locations[obj.symbol]):
                # if location
                if self.art[location] != obj.symbol:
                    #  respawning probability
                    if self._rng.random() < obj.eps_respawn:
                        self._object_locations[obj.symbol][i] = self.spawn(obj.symbol, location)
        
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

        bg_hight = self.art.shape[1] * (grid_size + gap_size) - gap_size
        bg_weight = self.art.shape[0] * (grid_size + gap_size) - gap_size
        bg_size = (bg_hight, bg_weight)
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

    def empty_point(self) -> Point:
        location = self.random_point()
        while self.art[location] != " ":
            location = self.random_point()

        return location

    def spawn(self, symbol: chr, location: Point  = None) -> Point:
        if location == None:
            location = self.empty_point()
        if symbol != 'P' and self._agent_location == location:
            return location

        self.art[location] = symbol
        return location

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



class RandomGridworld(Gridworld):
    def _get_observation(self) -> Any:
        return np.stack([np.where(self.art == x.symbol, 1, 0) for x in self.objects])

    def observation_spec(self) -> specs.BoundedArray:
        return specs.Array((len(self.objects), *self.shape), dtype=np.int32, name="observation")

    def spawn(self, symbol: chr, location: Point = None) -> Point:
        location = self.empty_point()
        if symbol != 'P' and self._agent_location == location:
            return location
        self.art[location] = symbol
        return location


class SimpleGridworld(TabularGridworld):
    def __init__(self, game_config, seed):
        self._location_map = {
            'P' : (1, 1),
            'a' : [(9, 8)],
            'b' : [(8, 1)],
            'c' : [(1, 3), (6, 9)],
        }
        super(SimpleGridworld, self).__init__(game_config, seed)


    def _reset(self) -> dm_env.TimeStep:
        self.art = self._art.copy()
        self._timestep = 0
    
        # spawn objects at random location
        for ch in "abc":
            self._object_locations[ch] = []
            for location in self._location_map[ch]:
                self._object_locations[ch].append(self.spawn(ch, location))
        # spawn agent at random location
        self._agent_location = self.spawn('P', self._location_map['P'])

        return dm_env.restart(self._get_observation())

SIMPLE = GridworldConfig(
    art=[
        "###########",
        "#    #    #",
        "#         #",
        "#    #    #",
        "#    #    #",
        "## ##### ##",
        "#    #    #",
        "#    #    #",
        "#         #",
        "#    #    #",
        "###########",
    ],
    objects=tuple(
        map(
            lambda x: GridworldObject(*x),
            [
                (1, 1.0, 1.0, 0.0, "a"),
                (1, 1.0, 0.0, 0.0, "b"),
                (2, -1.0, 1.0, 0.0, "c"),
            ],
        )
    ),
    max_steps=100,
)


DENSE = GridworldConfig(
    art=[
        "#############",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#############",
    ],
    objects=tuple(
        map(
            lambda x: GridworldObject(*x),
            [
                (2, 1.0, 0.0, 0.05, "a"),
                (1, -1.0, 0.5, 0.1, "b"),
                (1, -1.0, 0.0, 0.5, "c"),
            ],
        )
    ),
    max_steps=500,
)

SPARSE = GridworldConfig(
    art=[
        "###############",
        "#             #",
        "#             #",
        "#             #",
        "#             #",
        "#             #",
        "#             #",
        "#             #",
        "#             #",
        "#             #",
        "#             #",
        "#             #",
        "#             #",
        "#             #",
        "###############",
    ],
    objects=tuple(
        map(
            lambda x: GridworldObject(*x),
            [(1, 1.0, 1.0, 0.0, "a"), (1, -1.0, 1.0, 0.0, "b")],
        )
    ),
    max_steps=50,
)

LONG_HORIZON = GridworldConfig(
    art=[
        "#############",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#           #",
        "#############",
    ],
    objects=tuple(
        map(
            lambda x: GridworldObject(*x),
            [(2, 1.0, 0.0, 0.01, "a"), (2, -1.0, 0.5, 1.0, "b")],
        )
    ),
    max_steps=1000,
)

LONGER_HORIZON = GridworldConfig(
    art=[
        "###########",
        "#    #    #",
        "#         #",
        "#    #    #",
        "#   ###   #",
        "#    #    #",
        "#         #",
        "#    #    #",
        "###########",
    ],
    objects=tuple(
        map(
            lambda x: GridworldObject(*x),
            [
                (2, 1.0, 0.1, 0.01, "a"),
                (5, -1.0, 0.8, 1.0, "b"),
            ],
        )
    ),
    max_steps=2000,
)

LONG_DENSE = GridworldConfig(
    art=[
        "#############",
        "#           #",
        "#     #     #",
        "#     #     #",
        "#     #     #",
        "### ##### ###",
        "#     #     #",
        "#     #     #",
        "#           #",
        "#     #     #",
        "#     #     #",
        "#     #     #",
        "#############",
    ],
    objects=tuple(
        map(
            lambda x: GridworldObject(*x),
            [
                (4, 1.0, 0.0, 0.005, "a"),
            ],
        )
    ),
    max_steps=2000,
)

SMALL = GridworldConfig(
    art=[
        "#########",
        "#       #",
        "#  #    #",
        "#       #",
        "#    #  #",
        "#       #",
        "#########",
    ],
    objects=tuple(
        map(
            lambda x: GridworldObject(*x),
            [
                (2, 1.0, 0.0, 0.05, "a"),
                (2, -1.0, 0.5, 0.1, "b"),
            ],
        )
    ),
    max_steps=500,
)

SMALL_SPARSE = GridworldConfig(
    art=[
        "#########",
        "#       #",
        "#       #",
        "#       #",
        "#       #",
        "#       #",
        "#########",
    ],
    objects=tuple(
        map(
            lambda x: GridworldObject(*x),
            [
                (1, 1.0, 1.0, 1.0, "a"),
                (2, -1.0, 1.0, 1.0, "b"),
            ],
        )
    ),
    max_steps=50,
)

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
                (1, 1.0, 0.0, 1.0, "a"),
            ],
        )
    ),
    max_steps=2000,
)

class GridMaps(NamedTuple):
    DENSE = DENSE
    SPARSE = SPARSE
    LONG_HORIZON = LONG_HORIZON
    LONGER_HORIZON = LONGER_HORIZON
    LONG_DENSE = LONG_DENSE
    SMALL = SMALL
    SMALL_SPARSE = SMALL_SPARSE
    VERY_DENSE = VERY_DENSE

    SIMPLE = SIMPLE












