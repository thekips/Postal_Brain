import os
import sys
from absl import app, flags

#local import
sys.path.append(os.getcwd())
from environments.world import World
from algorithms.actor_critic.a2c_act import A2C, PolicyValueNet
from algorithms import experiment

# Environment.
flags.DEFINE_integer('max_steps', 1000, 'steps for agent to try in the environment')
flags.DEFINE_integer('seed', 42, 'seed for random number generation')
flags.DEFINE_integer('n_action', 9, 'total type of action to choose')
flags.DEFINE_integer('num_episodes', None, 'Overrides number of training eps')

# Agent.
flags.DEFINE_integer('num_hidden_layers', 2, 'number of hidden layers')
flags.DEFINE_integer('num_units', 64, 'number of units per hidden layer')
flags.DEFINE_integer('sequence_length', 32, 'mumber of transitions to batch')
flags.DEFINE_float('learning_rate', 1e-2, 'the learning rate')
flags.DEFINE_float('discount', .99, 'discounting on the agent side')

FLAGS = flags.FLAGS

def run(_):
    '''Runing experiment in a2c(output the action) by pytorch...'''

    env = World(FLAGS.max_steps, FLAGS.discount, FLAGS.seed, FLAGS.n_action)

    vit_odim = 64
    agent = A2C(
        obs_spec=env.observation_spec(),
        action_spec=env.action_spec(),
        max_sequence_length=FLAGS.sequence_length,
        network=PolicyValueNet((640, 480), (80, 80), vit_odim, env.action_spec()),
        learning_rate=FLAGS.learning_rate,
        discount=FLAGS.discount
    )

    num_episodes = FLAGS.num_episodes or env.bsuite_num_episodes
    experiment.run(agent, env, num_episodes, 'res/a2c_act.pkl')

if __name__ == '__main__':
    app.run(run)

#%%
import os
import sys
from absl import app, flags

#local import
sys.path.append('/home/ekips/Documents/Python/Postal_Brain/')
from environments.world import World
from algorithms.actor_critic.a2c_act import A2C, PolicyValueNet
from algorithms import experiment

env = World(1000, .99, 42, 9)
vit_odim = 64
agent = A2C(
    obs_spec=env.observation_spec(),
    action_spec=env.action_spec(),
    max_sequence_length=32,
    network=PolicyValueNet((600, 400), (100, 100), vit_odim, env.action_spec()),
    learning_rate=1e-2,
    discount=.99
)

num_episodes = None or env.bsuite_num_episodes

#%%
experiment.run(agent, env, num_episodes, 'res/a2c_act.pkl')

# %%
