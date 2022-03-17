import os
import sys
import numpy as np
import time
import torch
from absl import app, flags
from itertools import count
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.getcwd())

# Local import
from environments.world import ABSWorld, RELWorld

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
flags.DEFINE_integer('model', 1, 'choose model')
flags.DEFINE_float('learning_rate', 1e-4, 'the learning rate')
flags.DEFINE_float('discount', .99, 'discounting on the agent side')
flags.DEFINE_string('comment', 'Train with A2C model.', 'comment you want to print.')

FLAGS = flags.FLAGS

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
    env = ABSWorld(FLAGS.max_steps, FLAGS.seed, FLAGS.n_action)

    if FLAGS.model == 1:
        from algorithms.dqn.d3qn import Agent
        cwriter = SummaryWriter('logs/dtrain' + str(int(time.time())))
    elif FLAGS.model == 2:
        from algorithms.dqn.vd3qn import Agent
        cwriter = SummaryWriter('logs/vtrain' + str(int(time.time())))
    elif FLAGS.model == 3:
        from algorithms.dqn.d3qn_v import Agent
        cwriter = SummaryWriter('logs/train_v' + str(int(time.time())))

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
    
    n_steps = 0

    for episode in range(FLAGS.num_episodes):
        obs = env.reset()
        state = obs2state(obs)

        # 记录 reward
        total_reward = 0.0
        for _ in count():
            n_steps += 1

            action = agent.select_action(state, mode=FLAGS.mode)
            
            obs, reward = env.step(action)
            total_reward += reward
            
            next_state = obs2state(obs)

            reward = torch.tensor([reward], device=device)

            agent.memorize(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state

            if n_steps > 100:
                agent.update()

            if n_steps % 100 == 0:
                break

        print('Total steps: {} \t Episode: {}/{} \t Mean reward: {}'.format(n_steps, episode+1, FLAGS.num_episodes, total_reward / 100))
        cwriter.add_scalar('Mean_Reward_' + agent.name, total_reward / 100, episode + 1)

        # Test.
        rewards = []
        max_step = env._col
        for i in range(5):
            obs = env.reset()
            step = 0
            while True:
                state = obs2state(obs)
                action = agent.select_action(state, mode='test')
                _, reward = env.step(action)

                if int(action) == 4 or step >= max_step:
                    rewards.append(reward)
                    break
                step += 1
        
        cwriter.add_scalar('Compare_Rewards' + agent.name, np.mean(rewards), episode)

    agent.save_model(agent.name + '_' + str(FLAGS.learning_rate) + '.pkl')
    env.close()
    
if __name__ == '__main__':
    app.run(run)
