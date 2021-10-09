import enum
from functools import reduce

from utils.typedef import Spec
import matplotlib.pyplot as plt
import dm_env
import numpy as np

from environments.base import Environment
from utils.env_info import env_info
from utils.heatmap import Image
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs')

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
        self.discount = discount

        self.bsuite_num_episodes = 10

        # private:
        self._without_draw_item = True
        # TODO(thekips): make this can be used.
        # self._art = np.array([list(x) for x in game_config.art])
        self._rng = np.random.RandomState(seed)
        self._timestep = 0
        
        # self._plot = plt.imshow(np.empty(self.shape))
        
        env_info.set_model(tsp_model)
        self._agent_loc = env_info.agent_loc
        self._object_loc = env_info.object_loc

        # self._old_cost = env_info.cal_cost(self._agent_loc)

        print('world init reset.')
        self._reset()

    def observation_spec(self) -> Spec:
        shape = self.observation.shape
        return Spec(shape, reduce(lambda x, y: x * y, shape), int, 'observation')

    def action_spec(self) -> Spec:
        return Spec((2, ), -1, int, 'action')

    def _get_observation(self) -> np.ndarray:
        # TODO(thekips): 1.make more agent when program can run. 2.make image to class static.
        # image = Image(self._agent_loc, self._object_loc)
        # print('Gen the kde plot.')
        image = Image(self._agent_loc[52900009], self._object_loc[52900009])
        self.observation = image.getHeatMap()
        # print('Gen the kde plot end.')

        return self.observation

    def _reset(self) -> dm_env.TimeStep:
        print("thkeips: I get into _reset.")
        # self.art = self._art.copy()
        self._timestep = 0
    
        # reset agent at initial location.
        print('world start get env_info about loc')
        self._agent_loc = env_info.agent_loc

        print('begin restart env of reset.')
        res = dm_env.restart(self._get_observation())
        print('end restart.')
        return res

    def _step(self, action) -> dm_env.TimeStep:
        print("thekips: action is:", action)
        self._timestep += 1

        # update agent
        # TODO(thekips): enable multi agent.
        for agent in self._agent_loc.keys():
            self._agent_loc[agent] = action
        
            # for agent in self._agent_loc.keys():
            #     reward = 0.0
            #     vector = Actions(action).vector()
            #     self._agent_loc[agent] = (
            #         self._agent_loc[agent][0] + vector[0],
            #         self._agent_loc[agent][1] + vector[1]
            #     )
            #     break

        # compute reward by the cost
        cost = env_info.cal_cost(self._agent_loc)
        # self._old_cost = cost
        print("thekips: the cost in step is:", cost)

        writer.add_scalar('Step_Cost', cost, self._timestep)

        # 增加到最大步数时结束
        if self._timestep == self.max_steps:
            return dm_env.termination(cost, self._get_observation())

        return dm_env.transition(cost, self._get_observation())

    def _draw_item(self):
        plt.clf()
        plt.imshow(self.observation)
        plt.show()

    def bsuite_info(self):
        return {}

