# Built-in Library
from re import I
from typing import Sequence, Dict

# External Imports.
import dm_env
from dm_env import specs
from torch.distributions import distribution
from torch.distributions.categorical import Categorical
from torch.nn.modules import dropout
import tree

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Internal Imports.
from algorithms import base
import algorithms.utils.sequence as sequence
from algorithms.vit import ViT

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('we will use ', device)


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
        self._optimizer = torch.optim.Adam(
            network.parameters(), lr=learning_rate)

        # Create windowed buffer for learning from trajectories.
        self._buffer = sequence.Buffer(
            obs_spec, action_spec, max_sequence_length)

    def __compute_returns(self, final_value, rewards, discounts):
        value = final_value
        returns = []
        for step in reversed(range(len(rewards))):
            value = rewards[step] + self._discount * value * discounts[step]
            returns.insert(0, value)
        return torch.tensor(returns)

    def _step(self, trajectory: sequence.Trajectory, step: int):
        """Do a batch of SGD on the actor + critic loss."""
        observations, actions, rewards, discounts = trajectory

        # Add dummy batch dimensions.
        rewards = torch.unsqueeze(rewards, dim=-1)  # [T, 1]
        discounts = torch.unsqueeze(discounts, dim=-1)  # [T, 1]
        # print('rewards shape is',rewards.shape)
        # print('discount shape is', discounts.shape)
        # observations = torch.unsqueeze(observations, dim=1)  # [T+1, 1, ...]

        # calculate values by the value network.
        observations, final_observation = observations[:-1], observations[-1]
        policies, values = self._network(observations)
        _, final_value = self._network(final_observation)
        # print('values shape', values.shape, 'final_value', final_value.shape)
        # print('policies', policies.shape)
        values = torch.squeeze(values)

        # calculate the log probility of actions.
        policies = F.softmax(policies, dim=1)

        dists = [Categorical(policy) for policy in policies]
        log_probs = torch.stack([dist.log_prob(actions[i]) for i, dist in enumerate(dists)])
        log_probs = torch.squeeze(log_probs)

        # calculate actual values by the trajectory.
        returns = self.__compute_returns(final_value, rewards, discounts)
        returns = torch.squeeze(returns).detach()

        advantage = returns - values

        # compute loss.
        # TODO(thekips): optimize loss.
        actor_loss = -(advantage.detach() * log_probs).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        # log to tensor board event.
        print('Loss is:', loss.item())

        # update parameter.
        self._network.train()
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._network.parameters(), 0.5)

        self._optimizer.step()

    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        """Selects actions according to the latest softmax policy."""
        # combine the agent's location with obj's location respectively.
        observation = torch.tensor(timestep.observation)
        # print('observation shape is', observation.shape)

        policy, _ = self._network(observation)
        dist = F.softmax(policy, dim=0)
        action = dist.multinomial(num_samples=1)

        return base.Action(action)

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: base.Action,
        new_timestep: dm_env.TimeStep,
        step: int,
    ):
        """Receives a transition and performs a learning update."""

        self._buffer.append(timestep, action, new_timestep)

        # When the batch is full, do a step of SGD.
        if self._buffer.full() or new_timestep.last():
            trajectory = self._buffer.drain()
            trajectory = tree.map_structure(torch.tensor, trajectory)
            self._step(trajectory, step)

class PolicyValueNet(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        hidden_size: int,
        action_spec: specs.DiscreteArray,
    ):
        super(PolicyValueNet, self).__init__()
        print('Vit init.')
        self._vit = ViT(image_size=image_size, patch_size=patch_size, num_classes=hidden_size,
                       dim=1024, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1)
        self._policy_head = nn.Linear(hidden_size, action_spec.num_values)
        self._value_head = nn.Linear(hidden_size, 1)

        print('Vit end init.')
        # self._fc_layer = nn.Sequential()
        # for i in range(len(hidden_sizes) - 1):
        #     self._fc_layer.add_module(
        #         'fc'+str(i), nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        #     self._fc_layer.add_module('relu'+str(i), nn.ReLU())

    def forward(self, x: torch.Tensor):
        x = x / 255 # take value in x into [0, 1] as float.
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 0)
        x = self._vit(x)   # x.shape turn from image_size to hidden_size.
        # x = einops.rearrange(x, 'i j k l -> i (j k l)')

        policies = self._policy_head(x)
        value = self._value_head(x)

        return policies, value

