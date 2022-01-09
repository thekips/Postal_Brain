import torch
import os
import sys
from absl import app, flags

#local import
sys.path.append(os.getcwd())
from environments.world import ABSWorld
from algorithms.actor_critic.a2c_loc import A2C, PolicyValueNet
from algorithms import experiment
from utils.gpn_tsp import Attention, LSTM, GPN

load_root = 'models/gpn/gpn_tsp2000.pt'
tsp_model = torch.load(load_root).cuda()

flags.DEFINE_string('comment', 'Train with A2C model.', 'comment you want to print.')

# Environment.
flags.DEFINE_integer('max_steps', 1000, 'steps for agent to try in the environment')
flags.DEFINE_integer('seed', 42, 'seed for random number generation')
flags.DEFINE_integer('n_action', 9, 'total type of action to choose')
flags.DEFINE_integer('num_episodes', None, 'overrides number of training eps')
flags.DEFINE_string('time', '', 'time to append in model\'s name')

# Agent.
flags.DEFINE_integer('num_hidden_layers', 2, 'number of hidden layers')
flags.DEFINE_integer('num_units', 64, 'number of units per hidden layer')
flags.DEFINE_integer('sequence_length', 32, 'mumber of transitions to batch')
flags.DEFINE_float('learning_rate', 1e-2, 'the learning rate')
flags.DEFINE_float('discount', .99, 'discounting on the agent side')

FLAGS = flags.FLAGS

def run(_):
    '''Runing experiment in a2c(output the action) by pytorch...'''
    print(FLAGS.comment)

    env = ABSWorld(FLAGS.max_steps, FLAGS.seed, FLAGS.n_action)

    vit_odim = 64
    agent = A2C(
        obs_spec=env.observation_spec(),
        action_spec=env.action_spec(),
        max_sequence_length=FLAGS.sequence_length,
        network=PolicyValueNet((600, 400), (100, 100), vit_odim),
        learning_rate=FLAGS.learning_rate,
        discount=FLAGS.discount
    )

    num_episodes = FLAGS.num_episodes or env.bsuite_num_episodes
    experiment.run(agent, env, num_episodes, 'models/actor_critic/a2c_act_'+ FLAGS.time + '.pkl')

if __name__ == '__main__':
    app.run(run)
