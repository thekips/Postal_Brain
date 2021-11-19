# Built-in Library
from re import I
from typing import Sequence, Dict
import time
# External Imports.
import dm_env
from dm_env import specs
from torch.utils.tensorboard import SummaryWriter
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

writer = SummaryWriter('logs')

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
        stime = time.time()
        observations, actions, costs, discounts = trajectory

        # calculate values by the value network.
        observations = observations[:-1]
        _, values = self._network(observations)
        values = torch.squeeze(values)

        # compute loss.
        advantage = costs - values
        print("costs shape:", costs.shape, "costs", costs, "values", values)
        actor_loss = costs.mean()
        critic_loss = advantage.pow(2).mean()
        loss = (actor_loss + 0.05 * critic_loss).pow(0.1)
        # print("actor_loss ", actor_loss, "critic_loss", critic_loss, "loss", loss)

        # log to tensor board event.
        writer.add_scalar('Multi-Step_Loss', loss.item())
        print('Multi-Step loss is:', loss.item())

        # update parameter.
        self._network.train()
        #TODO(thekips): test.
        for p in self._network._policy_head.parameters():
            p.requires_grad = False
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._network.parameters(), 0.5)

        self._optimizer.step()
        print("Time of step use %ds" % (time.time() - stime))

    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        """Selects actions according to the latest softmax policy."""
        # combine the agent's location with obj's location respectively.
        observation = torch.tensor(timestep.observation)
        # print('observation shape is', observation.shape)

        policies, _ = self._network(observation)
        return policies

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
    ):
        super(PolicyValueNet, self).__init__()
        print('Vit init.')
        self._vit = ViT(image_size=image_size, patch_size=patch_size, num_classes=hidden_size,
                       dim=1024, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1)
        self._policy_head = nn.Linear(hidden_size, 2)
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
        # (thekips):When use pytorch<=1.3.0, you should use x.float() to avoid scalar type error.
        x = self._vit(x.float())   # x.shape turn from image_size to hidden_size.
        # x = einops.rearrange(x, 'i j k l -> i (j k l)')

        policies = self._policy_head(x)
        value = self._value_head(x)

        return policies.squeeze().tolist(), value