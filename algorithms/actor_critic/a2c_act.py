# External Imports.
import time
from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F

# Internal Imports.
from algorithms import base
import algorithms.utils.sequence as sequence
from algorithms.vit import ViT
from algorithms.utils.memory import ReplayMemory, Transition
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='logs/a2c_v_loss' + str(int(time.time())))

class A2C(base.Agent):
    """A simple TensorFlow-based feedforward actor-critic implementation."""

    def __init__( self, obs_spec, action_spec, discount, learning_rate, device, mem_size, batch_size, patch_size):
        """A simple actor-critic agent."""

        self._discount = discount
        self.device = device
        self.batch_size = batch_size
        self.memory = ReplayMemory(mem_size)

        # Internalise network and optimizer.
        self._network = PolicyValueNet((obs_spec[0], obs_spec[1]), patch_size, action_spec).to(self.device)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=learning_rate)
        
        self.name = 'a2c_v'
        print(self.name + ' inited...')
        
        # Create windowed buffer for learning from trajectories.
        self.n_steps = 0

    def __compute_returns(self, final_value, rewards):
        value = final_value
        returns = []
        for step in reversed(range(len(rewards))):
            value = rewards[step] + self._discount * value
            returns.insert(0, value)
        return torch.tensor(returns).to(self.device)

    def _step(self):
        """Do a batch of SGD on the actor + critic loss."""
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        actions = tuple((map(lambda a: torch.tensor([[a]], device=self.device), batch.action)))
        rewards = tuple((map(lambda r: torch.tensor([r], device=self.device), batch.reward)))
        states = torch.cat(batch.state).to(self.device)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)

        # calculate values by the value network.
        policies, values = self._network(states)

        values = torch.squeeze(values)

        _, final_value = self._network(states[-1])
        returns = self.__compute_returns(final_value, rewards)
        returns = torch.squeeze(returns).detach()

        advantage = returns - values

        # TODO(thekips): optimize loss.
        # calculate the log probility of actions.
        policies = F.softmax(policies, dim=1)

        dists = [Categorical(policy) for policy in policies]
        log_probs = torch.stack([dist.log_prob(actions[i]) for i, dist in enumerate(dists)])
        log_probs = torch.squeeze(log_probs)

        actor_loss = -(advantage.detach() * log_probs).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        # log to tensor board event.
        writer.add_scalar('Loss_Step', loss.item(), self.n_steps)

        # update parameter.
        self._network.train()
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._network.parameters(), 0.5)

        self._optimizer.step()

    def select_action(self, state, mode='train') -> base.Action:
        if mode == 'train':
            self.n_steps += 1
        # print('observation shape is', observation.shape)

        with torch.no_grad(): 
            policy, _ = self._network(state.to(self.device))
            dist = F.softmax(policy, dim=0)
            action = dist.multinomial(num_samples=1)

            return action

    def update(self):
        """Receives a transition and performs a learning update."""
        if len(self.memory) < self.batch_size:
            return

        self._step()

    def memorize(self, state, action, state_next, reward):
        self.memory.push(state, action, state_next, reward)

class PolicyValueNet(nn.Module):
    def __init__( self, image_size, patch_size, action_spec):
        super(PolicyValueNet, self).__init__()
        vit_out_dim = 128
        self._vit = ViT(image_size=image_size, patch_size=patch_size, num_classes=vit_out_dim,
                       dim=1024, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1)
        self._policy_head = nn.Linear(vit_out_dim, action_spec)
        self._value_head = nn.Linear(vit_out_dim, 1)

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

