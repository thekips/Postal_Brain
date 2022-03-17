import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='logs/dloss' + str(int(time.time())))

#Local Import
from algorithms.utils.memory import ReplayMemory, Transition

class DuelingDQN(nn.Module):
    def __init__(self, in_dims, in_channels=3, n_actions=9):
        """
        Initialize Deep Q Network

        :param in_channels (int): number of input channels
        :param n_actions (int): number of outputs
        """
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_out_size(input, kernel_size, stride, padding=0):
            return (input + padding * 2 - kernel_size) // stride + 1
        convw = conv2d_out_size(conv2d_out_size(conv2d_out_size(in_dims[0], 7, 4), 3, 2), 3, 1)
        convh = conv2d_out_size(conv2d_out_size(conv2d_out_size(in_dims[1], 7, 4), 3, 2), 3, 1)

        self.fc4 = nn.Linear(64 * convw * convh, 512)

        self.fc5 = nn.Linear(512, 128)
        self.head_val = nn.Linear(128, 1)   # V值
        self.head_adv = nn.Linear(128, n_actions)   # Advance值
        
    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        x = F.relu(self.fc5(x))
        adv = self.head_adv(x)
        val = self.head_val(x).expand(-1, adv.size(1))
        out = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))

        return out

class Agent:
    '''
    Use [Dueling DQN] network to define a [Double Duelling DQN Agent].
    '''
    def __init__(self, discount, lr, input_dims, in_channels, n_actions, mem_size, batch_size,
                  replace, device):

        self.discount = discount
        self.lr = lr
        self.n_actions = n_actions
        
        self.memory = ReplayMemory(mem_size)

        self.batch_size = batch_size
        self.replace = replace
        self.device = device

        self.main_net = DuelingDQN(in_dims=input_dims, in_channels=in_channels, n_actions=self.n_actions).to(self.device)
        self.target_net = DuelingDQN(in_dims=input_dims, in_channels=in_channels, n_actions=self.n_actions).to(self.device)

        self.target_net.load_state_dict(self.main_net.state_dict())

        # print(self.main_net)
        self.name = 'D3QN'
        print(self.name + ' inited...')

        self.optimizer = optim.Adam(self.main_net.parameters(), lr=lr)

        self.n_steps = 0

    def select_action(self, state, mode='train'):
        """
        :param steps: 用来更新eposilon的步数, 可以是episode
        :type steps: int
        """
        if mode == 'train':
            self.n_steps += 1
            if self.n_steps % self.replace == 0:
                self.update_target()

        epsilon = (1 / (self.n_steps / 10 + 1))

        if epsilon <= np.random.uniform(0, 1) or mode == 'test':
            self.main_net.eval()
            with torch.no_grad():
                action = self.main_net(state.to('cuda')).max(1)[1].view(1, 1)
                # _action = self.main_net(state.to('cuda'))
                # print(_action)
                # action = _action.max(1)[1].view(1, 1)
                # print(action)

        else:
            action = torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

        return action

    def get_qvalue(self, state):
        self.main_net.eval()
        with torch.no_grad():
            qvalue = self.main_net(state.to('cuda'))
        return qvalue

    def update(self):
        """
        经验回放更新网络参数
        """
        # 检查经验池数据量是否够一个批次
        if len(self.memory) < self.batch_size:
            return

        # 创建小批量数据
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()

        self.expected_state_action_values = self.get_expected_state_action_values()

        self.update_main()

    def make_minibatch(self):
        """创建小批量数据"""

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        actions = tuple((map(lambda a: torch.tensor([[a]], device=self.device), batch.action)))
        rewards = tuple((map(lambda r: torch.tensor([r], device=self.device), batch.reward)))

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        """获取期望的Q值"""

        self.main_net.eval()
        self.target_net.eval()

        self.state_action_values = self.main_net(self.state_batch).gather(1, self.action_batch)

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, self.batch.next_state)),
            device=self.device, dtype=torch.bool)

        next_state_values = torch.zeros(self.batch_size, device=self.device)

        next_state_values[non_final_mask] = self.target_net(self.non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.discount) + self.reward_batch
        
        return expected_state_action_values

    def update_main(self):
        """更新网络参数"""

        # 将网络切换训练模式
        self.main_net.train()

        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))
        # print("loss is:", loss)
        # print(self.state_action_values.squeeze())
        # print(self.expected_state_action_values)
        writer.add_scalar('Loss_Step', loss.item(), self.n_steps)
        self.optimizer.zero_grad()

        loss.backward()

        for param in self.main_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.main_net.state_dict())

    def save_model(self, filename):
        torch.save(self.main_net, filename)

    def load_model(self, filename):
        self.main_net = torch.load(filename)
        self.target_net.load_state_dict(self.main_net.state_dict())
    
    def memorize(self, state, action, state_next, reward):
        self.memory.push(state, action, state_next, reward)
