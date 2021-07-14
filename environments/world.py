import enum
import matplotlib.pyplot as plt
import dm_env
import numpy as np
from dm_env import specs

from environments.base import Environment
from utils.env_info import env_info
from utils.heatmap import Image

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

class World(Environment):
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

    def _get_observation(self) -> np.ndarray:
        # TODO(thekips): make more agent when program can run.
        # image = Image(self._agent_loc, self._object_loc)
        image = Image(self._agent_loc[52900009], self._object_loc[52900009])
        self.observation = image.getHeatMap()

        return self.observation

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

    def _draw_item(self):
        plt.clf()
        plt.imshow(self.observation)
        plt.show()

    def bsuite_info(self):
        return {}

