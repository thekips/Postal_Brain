import torch
import time
import os
import sys
from absl import app, flags
import numpy as np
from itertools import count

#local import
sys.path.append(os.getcwd())
from environments.world import ABSWorld
from algorithms.actor_critic.a2c_act import A2C, PolicyValueNet
from torch.utils.tensorboard import SummaryWriter
from utils.gpn_tsp import Attention, LSTM, GPN

# load_root = 'models/gpn/gpn_tsp2000.pt'
# tsp_model = torch.load(load_root).cuda()

flags.DEFINE_string('comment', 'Train with A2C model.', 'comment you want to print.')

# Environment.
flags.DEFINE_integer('max_steps', 500, 'steps for agent to try in the environment')
flags.DEFINE_integer('seed', 42, 'seed for random number generation')
flags.DEFINE_integer('n_action', 9, 'total type of action to choose')
flags.DEFINE_integer('num_episodes', 500, 'overrides number of training eps')
flags.DEFINE_string('time', '', 'time to append in model\'s name')
flags.DEFINE_string('mode', 'train', 'ways to use model: train or test')

# Agent.
flags.DEFINE_integer('num_hidden_layers', 2, 'number of hidden layers')
flags.DEFINE_integer('num_units', 64, 'number of units per hidden layer')
flags.DEFINE_integer('mem_size', 64, 'limit of memory')
flags.DEFINE_integer('batch_size', 32, 'mumber of transitions to batch')
flags.DEFINE_float('learning_rate', 1e-2, 'the learning rate')
flags.DEFINE_float('discount', .99, 'discounting on the agent side')

FLAGS = flags.FLAGS
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('we will use ', device)

def obs2state(obs):
    """ 
    观察值转换成状态
    """
    state = np.array(obs)
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

def run(_):
    '''Runing experiment in a2c(output the action) by pytorch...'''
    print(FLAGS.comment)

    env = ABSWorld(FLAGS.max_steps, FLAGS.seed, FLAGS.n_action)

    cwriter = SummaryWriter('logs/drun' + str(int(time.time())))
    model_name = 'a2c_v_'

    patch_size = (80, 80)
    obs_spec = env.observation_spec()
    agent = A2C(
        obs_spec=env.observation_spec(),
        action_spec=env.action_spec(),
        discount=FLAGS.discount,
        learning_rate=FLAGS.learning_rate,
        device = device,
        mem_size=FLAGS.mem_size,
        batch_size=FLAGS.batch_size,
        patch_size=patch_size
    )

    n_steps = 0
    for episode in range(FLAGS.num_episodes):
        obs = env.reset()
        state = obs2state(obs)

        # 记录 reward
        total_reward = 0.0
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

            if n_steps > 32:
                agent.update()

            if n_steps % 100 == 0:
                break

        print('Total steps: {} \t Episode: {}/{} \t Mean reward: {}'.format(n_steps, episode+1, FLAGS.num_episodes, total_reward / 100))
        cwriter.add_scalar('Mean_Reward_' + agent.name, total_reward / 100, episode + 1)
        
    env.close()
    agent.save_model(model_name + str(FLAGS.learning_rate) + '.pkl')
    return

if __name__ == '__main__':
    app.run(run)
