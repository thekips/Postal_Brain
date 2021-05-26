# Built-in Library
from re import I
from typing import OrderedDict, Sequence, Dict

# External Imports.
import dm_env
from dm_env import specs
from torch.distributions.categorical import Categorical
import tree

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Internal Imports.
from algorithms import base
from algorithms.utils import sequence

class A2C(base.Agent):
    """A simple TensorFlow-based feedforward actor-critic implementation."""

    def __init__(
            self,
            obs_spec: specs.Array,
            action_spec: specs.Array,
            max_sequence_length: int,
            network: 'PolicyValueNet',
            learning_rate: float,
            discount: float,
    ):
        """A simple actor-critic agent."""

        self._discount = discount

        # Internalise network and optimizer.
        self._network = network
        self._optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

        # Create windowed buffer for learning from trajectories.
        self._buffer = sequence.Buffer(obs_spec, action_spec, max_sequence_length)

    def __compute_returns(self, final_value, rewards, discounts):
        value = final_value
        returns = []
        for step in reversed(range(len(rewards))):
            value = rewards[step] + self._discount * value * discounts[step]
            returns.insert(0, value)
        return returns


    def _sample_policy(self, inputs: torch.tensor) -> torch.tensor:
        ''' 这个函数为 select_action 准备 '''
        policy, _ = self._network(inputs)  # TODO(thekips): should be self._network.forward(inputs)
        dist = F.softmax(policy, dim=1)
        action = dist.multinomial(num_samples=1)

        return action

    def _step(self, trajectory: sequence.Trajectory):
        """Do a batch of SGD on the actor + critic loss."""
        observations, actions, rewards, discounts = trajectory

        # Add dummy batch dimensions.
        rewards = torch.unsqueeze(rewards, dim=-1)  # [T, 1]
        discounts = torch.unsqueeze(discounts, dim=-1)  # [T, 1]
        observations = torch.unsqueeze(observations, dim=1)  # [T+1, 1, ...]

        # calculate values by the value network.
        observations, final_observation = observations[:-1], observations[-1]
        policies, values = self._network(observations)
        _, final_value = self._network(final_observation)
        values = torch.cat(values)

        # calculate the log probility of actions.
        dists = Categorical(policies)
        log_probs = dists.log_prob(actions)
        log_probs = torch.cat(log_probs)

        # calculate actual values by the trajectory. 
        returns = self.__compute_returns(final_value, rewards, discounts)
        returns = torch.cat(returns).detach()

        advantage = returns - values

        # compute loss.
        actor_loss = -(advantage.detach() * log_probs).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        # update parameter.
        self._network.train()
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._network.parameters(), 0.5)

        self._optimizer.step()

    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        """Selects actions according to the latest softmax policy."""
        obs_tensor = torch.tensor(timestep.observation, dtype=torch.double)
        # 可能是增加一维，以备网络的批处理
        observation = torch.unsqueeze(obs_tensor, dim=0)
        policy, _ = self._network(observation)  # TODO(thekips): should be self._network.forward(inputs)
        dist = F.softmax(policy, dim=1)
        action = dist.multinomial(num_samples=1)

        return action.numpy()

    def update(
            self,
            timestep: dm_env.TimeStep,
            action: base.Action,
            new_timestep: dm_env.TimeStep,
    ):
        """Receives a transition and performs a learning update."""

        self._buffer.append(timestep, action, new_timestep)

        # When the batch is full, do a step of SGD.
        if self._buffer.full() or new_timestep.last():
            trajectory = self._buffer.drain()
            trajectory = tree.map_structure(torch.tensor, trajectory)
            self._step(trajectory)


class PolicyValueNet(nn.Module):
    """A simple conv neural networks with a value and a policy head."""

    def __init__(
            self,
            obs_spec: specs.Array,
            action_spec: specs.DiscreteArray,
            hidden_sizes: Dict[str, Sequence[int]] = {'Conv': (16,), 'Dense': (32,)}
    ):

        super(PolicyValueNet, self).__init__()
        conv_Layers = hidden_sizes.get('Conv', tuple())
        dense_sizes = hidden_sizes.get('Dense', tuple())

        # 左端的卷积层
        self.conv_layer = nn.Sequential()
        D, H, W = obs_spec.shape

        H -= len(conv_Layers) * 2
        W -= len(conv_Layers) * 2

        conv_Layers = [D] + conv_Layers

        for i in range(len(conv_Layers)-1):
            self.conv_layer.add_module('conv' + str(i),
                                       nn.Conv2d(
                in_channels=conv_Layers[i],
                out_channels=conv_Layers[i+1],
                kernel_size=3
            )
            )
            self.conv_layer.add_module('relu', nn.ReLU())

        self.hidden_size = conv_Layers[-1] * H * W

        # 中间的全连接层
        self.fc_layer = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(self.hidden_size, dense_sizes[0])),
            ('relu', nn.ReLU())
        ]))

        for i in range(len(dense_sizes)-1):
            self.fc_layer.add_module('fc'+str(i+1), nn.Linear(dense_sizes[i], dense_sizes[i+1]))
            self.fc_layer.add_module('relu', nn.ReLU())

        # 网络的右端
        self._policy_head = nn.Linear(dense_sizes[-1], action_spec.num_values)
        self._value_head = nn.Linear(dense_sizes[-1], 1)

        print(self.conv_layer)
        print(self.fc_layer)

    def forward(self, inputs):

        x = self.conv_layer(inputs)
        x.view(x[0], -1)
        x = self.fc_layer(x)
        value = self._value_head(x)
        policy = self._policy_head(x)

        return policy, value

    def num_flat_features(self, x):
        size = x.size()[1:]

def default_agent(
        obs_spec: specs.Array,
        action_spec: specs.DiscreteArray,
        hidden_sizes: Dict[str, Sequence[int]] = {'Conv': (16), 'Dense': (32)}
) -> base.Agent:
    """Initialize a DQN agent with default parameters."""
    network = PolicyValueNet(
        hidden_sizes=hidden_sizes,
        action_spec=action_spec,
    )

    return A2C(
        obs_spec=obs_spec,
        action_spec=action_spec,
        network=network,
        learning_rate=3e-3,
        max_sequence_length=32,
        td_lambda=0.9,
        discount=0.99,
        seed=42,
    )
