import os
import sys
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import time
import torch
from absl import app, flags
from itertools import count
from torch.jit import annotate
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.getcwd())

# Local Import
from environments.world import ABSWorld, RELWorld

# Environment.
flags.DEFINE_integer('max_steps', 1000, 'steps for agent to try in the environment')
flags.DEFINE_integer('seed', 42, 'seed for random number generation')
flags.DEFINE_integer('n_action', 9, 'total type of action to choose')
flags.DEFINE_integer('num_episodes', 200, 'train all.')
flags.DEFINE_string('time', '', 'time to append in model\'s name')
flags.DEFINE_string('mode', 'train', 'ways to use model: train or test')

# Agent.
flags.DEFINE_integer('num_hidden_layers', 2, 'number of hidden layers')
flags.DEFINE_integer('num_units', 64, 'number of units per hidden layer')
flags.DEFINE_integer('mem_size', 10000, 'limit of memory')
flags.DEFINE_integer('batch_size', 32, 'mumber of transitions to batch')
flags.DEFINE_integer('replace', 1000, 'Interval for replacing target network')
flags.DEFINE_integer('model', 1, 'choose model')
flags.DEFINE_float('discount', .99, 'discounting on the agent side')
flags.DEFINE_float('learning_rate', 1e-4, 'the learning rate')
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
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

def run(_):
    env = ABSWorld(FLAGS.max_steps, FLAGS.seed, FLAGS.n_action)

    if FLAGS.mode == 'draw':
        env._draw_item(pos=(7,6) ,mode='heatmap', is_patch=False)
        return

    if FLAGS.model == 1:
        from algorithms.dqn.d3qn import Agent
        model_name = 'd3qn_'
        if FLAGS.mode == 'train':
            cwriter = SummaryWriter('logs/drun' + str(int(time.time())))
    elif FLAGS.model == 2:
        from algorithms.dqn.vd3qn import Agent
        model_name = 'vd3qn_'
        if FLAGS.mode == 'train':
            cwriter = SummaryWriter('logs/vrun' + str(int(time.time())))
    elif FLAGS.model == 3:
        from algorithms.dqn.d3qn_v import Agent
        model_name = 'd3qn_v_'
        if FLAGS.mode == 'train':
            cwriter = SummaryWriter('logs/run_v' + str(int(time.time())))

    agent = Agent(
        discount=FLAGS.discount,
        lr=FLAGS.learning_rate,
        input_dims=env.observation_spec(),
        n_actions=env.action_spec(),
        in_channels=3,
        mem_size=FLAGS.mem_size,
        batch_size=FLAGS.batch_size,
        replace=FLAGS.replace,
        device=device
    )

    if FLAGS.mode == 'visual':
        rvalue = [[0 for i in range(10)] for j in range(10)]
        for i in range(10):
            for j in range(10):
                env._agent_loc = (i, j)
                state = obs2state(env._get_observation())
                qvalue = agent.get_qvalue(state)


                rvalue[9 - i][j] = float(1e8 * qvalue.max())

        agent.load_model(model_name + str(FLAGS.learning_rate) + '.pkl')
        svalue = [[0 for i in range(10)] for j in range(10)]
        for i in range(10):
            for j in range(10):
                env._agent_loc = (i, j)
                state = obs2state(env._get_observation())
                qvalue = agent.get_qvalue(state)


                svalue[9 - i][j] = float(1e8 * qvalue.max())

        plt.subplot(1, 3, 1)
        sns.heatmap(svalue, cmap='Greys', cbar=False)
        plt.savefig(model_name + '_value.svg')
        plt.show()
        return

    if FLAGS.mode == 'debug':
        agent.load_model(model_name + str(FLAGS.learning_rate) + '.pkl')
        torch.set_printoptions(profile='full')
        step_debug = input("Please input whether step(y/n):")
        posit = input("input the tuple: (")
        env._agent_loc = eval('(' + posit + ')')
        state = obs2state(env._get_observation(isplot=True))

        while True:
            action = agent.select_action(state, mode='test')
            _, reward = env.step(action)
            print(reward)
            state = obs2state(env._get_observation(isplot=True))

            if step_debug == 'y':
                os.system('read -n 1')
    
    if FLAGS.mode == 'test':
        agent.load_model(model_name + str(FLAGS.learning_rate) + '.pkl')
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

    if FLAGS.mode == 'trans':
        print('Use previous mode to transfer.')
        agent.load_model(model_name + str(FLAGS.learning_rate) + '.pkl')

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

            if n_steps > 100:
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
