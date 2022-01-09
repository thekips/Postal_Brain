import argparse
from itertools import count
import math
import random
import numpy as np
import time
import torch

import sys
from absl import app, flags

#local import
import os
sys.path.append(os.getcwd())
from algorithms.dqn.wrappers import make_env
from environments.world import ABSWorld, RELWorld
# from utils.gpn_tsp import Attention, LSTM, GPN

load_root = 'models/gpn/gpn_tsp2000.pt'
# tsp_model = torch.load(load_root).cuda()

flags.DEFINE_string('comment', 'Train with A2C model.', 'comment you want to print.')

# Environment.
flags.DEFINE_integer('max_steps', 1000, 'steps for agent to try in the environment')
flags.DEFINE_integer('seed', 42, 'seed for random number generation')
flags.DEFINE_integer('n_action', 9, 'total type of action to choose')
flags.DEFINE_integer('num_episodes', 1000, 'train all.')
flags.DEFINE_string('time', '', 'time to append in model\'s name')
flags.DEFINE_string('mode', 'train', 'ways to use model: train or test')

# Agent.
flags.DEFINE_integer('num_hidden_layers', 2, 'number of hidden layers')
flags.DEFINE_integer('num_units', 64, 'number of units per hidden layer')
flags.DEFINE_integer('mem_size', 10000, 'limit of memory')
flags.DEFINE_integer('batch_size', 32, 'mumber of transitions to batch')
flags.DEFINE_integer('replace', 1000, 'Interval for replacing target network')

flags.DEFINE_float('learning_rate', 1e-4, 'the learning rate')
flags.DEFINE_float('discount', .99, 'discounting on the agent side')

FLAGS = flags.FLAGS

# from algorithms.dqn.d3qn import Agent
from algorithms.dqn.d3qn import Agent

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('we will use ', device)

def obs2state(obs):
    """ 
    观察值转换成状态
    """
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

def run(_):
    env = ABSWorld(FLAGS.max_steps, None, FLAGS.seed, FLAGS.n_action)
    FLAGS.mode = 'draw'
    if FLAGS.mode == 'draw':
        env._draw_item(pos=(7,6) ,mode='heatmap', is_patch=False)
        return

    agent = Agent(
        discount=FLAGS.discount,
        lr=FLAGS.learning_rate,
        input_dims=env.observation_spec(),
        in_channels=3,
        n_actions=env.action_spec(),
        mem_size=FLAGS.mem_size,
        batch_size=FLAGS.batch_size,
        replace=FLAGS.replace,
        device=device
    )
    if FLAGS.mode == 'debug':
        agent.load_model('dqn3_' + str(FLAGS.learning_rate) + '.pkl')
        torch.set_printoptions(profile='full')
        step_debug = input("Please input whether step(y/n):")
        posit = input("input the tuple: (")
        env._agent_loc = eval('(' + posit + ')')
        state = obs2state(env._get_observation(isplot=True))

        while True:

            # Q = agent.main_net(state.to('cuda'))
            # print(Q)

            action = agent.select_action(state, mode='test')
            _, reward = env.step(action)
            print(reward)
            state = obs2state(env._get_observation(isplot=True))

            if step_debug == 'y':
                os.system('read -n 1')
    
    if FLAGS.mode == 'test':
        agent.load_model('dqn3_' + str(FLAGS.learning_rate) + '.pkl')
        state = obs2state(env._get_observation(isplot=True))
        while True:

            Q = agent.main_net(state.to('cuda'))
            print(Q)

            action = agent.select_action(state, mode=FLAGS.mode)
            _, reward = env.step(action)
            print(reward)
            state = obs2state(env._get_observation(isplot=True))
            time.sleep(0.4)

            if int(action) == 4:
                print('reset env')
                time.sleep(2)
                print(env._agent_loc)
                env.reset()

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('logs/run')

    if FLAGS.mode == 'trans':
        print('thekips: use previous mode to transfer.')
        agent.load_model('dqn3_' + str(FLAGS.learning_rate) + '.pkl')

    mean_rewards = []
    episode_reward = []
    n_steps = 0

    for episode in range(FLAGS.num_episodes):
        obs = env.reset()
        state = obs2state(obs)

        # 记录 reward
        total_reward = 0.0
        stay_cnt = 0
        old_reward = 0
        flag = False
        for t in count():
            n_steps += 1

            action = agent.select_action(state, mode=FLAGS.mode)
            
            # print('agent loc and mean is', env._agent_loc, np.array(state).mean()) 
            obs, reward = env.step(action)
            # print('action is ', action, 'reward is ', reward)
            total_reward += reward
            
            next_state = obs2state(obs)

            reward = torch.tensor([reward], device=device)

            agent.memorize(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state

            if n_steps > 100:
                agent.update()

            if n_steps % 100 == 0:
                break
            # stay_cnt = stay_cnt + 1 if old_reward == reward else 0
            # old_reward = reward
            # if stay_cnt == 10:
            #     print("10 steps stay a pos")
            #     break

        
        print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(n_steps, episode+1, FLAGS.num_episodes, total_reward))
        

        # 计算平均 reward
        episode_reward.append(total_reward)
        mean_100ep_reward = round(np.mean(episode_reward[-100:]), 1)
        writer.add_scalar('Mean_Reward' + agent.name, mean_100ep_reward, episode)
        
    env.close()
    
    # 保存模型
    agent.save_model('dqn3_' + str(FLAGS.learning_rate) + '.pkl')
    return

if __name__ == '__main__':
    app.run(run)
