from typing import Any, Dict, NamedTuple, Tuple

import dm_env
from dm_env import specs
import numpy as np
from lrla.environments import base


class DelayedchainConfig(NamedTuple):
    chain_length_range: Tuple[int, int]
    noisy: bool


class Delayedchain(base.Environment):
    def __init__(self, game_config: DelayedchainConfig, seed: int = None):
        super().__init__()
        # 用来随机机器
        self._rng = np.random.RandomState(seed)
        # public:
        self.chain_length = self._rng.choice(game_config.chain_length_range)

        self.noisy = game_config.noisy

        # private
        self._timestep = 0
        self._has_umbrella = 0
        self._reset()
        
        # YUANYIN modify
        self._total_regret = 0

    def _get_observation(self):
        return self._has_umbrella * self.chain_length + self._timestep

    def _reset(self) -> dm_env.TimeStep:
        self._timestep = 0
        self._raining = self._rng.binomial(1, 0.5)
        self._has_umbrella = self._rng.binomial(1, 0.5)
        return dm_env.restart(self._get_observation())

    def _step(self, action: int) -> dm_env.TimeStep:
        self._timestep += 1
        reward = 0
        if self._timestep == 1:  # you can only pick up umbrella t=1
            self._has_umbrella = action

        if self._timestep == self.chain_length:  # reward only at end.
            if self._has_umbrella == self._raining:
                reward = 1.0
            else:
                reward = -1.0
                self._total_regret += 2.0
            observation = self._get_observation()
            return dm_env.termination(reward=reward, observation=observation)

        if self.noisy:
            reward = 2.0 * self._rng.binomial(1, 0.5) - 1.0
        return dm_env.transition(reward, self._get_observation())

    def action_spec(self):
        return specs.DiscreteArray(2, name="action")

    def observation_spec(self):
        return specs.DiscreteArray(self.chain_length * 2, name="observation")

    def bsuite_info(self) -> Dict[str, Any]:
        return {}

    def render(self, mode="ansi"):
        if mode == "ansi":
            weather = "-" if self._raining else "~"
            baseline = weather * self.chain_length
            pawn = "☂" if self._has_umbrella else "👤"
            walked = baseline.replace(weather, pawn, self._timestep)
            print(walked)
        return


class StateDistraction(Delayedchain):
    def __init__(self, game_config: DelayedchainConfig, seed: int = None):
        self.is_correct_action = 1
        self.chosen_correct_action = 1
        super(StateDistraction, self).__init__(game_config, seed)


    def _get_observation(self):
        import random
        obs1 = [self.is_correct_action, self.chosen_correct_action]
        obs2 = [random.randint(0, 1) for _ in range(20)]
        return obs1 + obs2

    def _step(self, action: int) -> dm_env.TimeStep:
        self._timestep += 1
        reward = 0
        
        if self._timestep == 1:  # you can only pick up umbrella t=1
            self._has_umbrella = action
            self.is_correct_action = int(self._has_umbrella == self._raining)

        if self._timestep == self.chain_length:  # reward only at end.
            if self._has_umbrella == self._raining:
                reward = 1.0
            else:
                reward = -1.0
                self._total_regret += 2.0
            observation = self._get_observation()
            return dm_env.termination(reward=reward, observation=observation)

        self.chosen_correct_action = int(action == self._raining)

        if self.noisy:
            reward = 2.0 * self._rng.binomial(1, 0.5) - 1.0
        return dm_env.transition(reward, self._get_observation())

    def observation_spec(self):
        return specs.Array((22,), dtype=np.int32, name="observation")



class DelayedchainMaps(NamedTuple):
    SHORT = DelayedchainConfig((5, 30), False)
    SHORT_AND_NOISY = DelayedchainConfig((5, 30), True)
    LONG = DelayedchainConfig((5, 50), False)
    LONG_AND_NOISY = DelayedchainConfig((5, 50), True)
    STATE_DISTRACTION = DelayedchainConfig((5, 30), False)
