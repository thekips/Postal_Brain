from absl import app, flags

#local import
from environments.world import World
from algorithms.pytorch.actor_critic.a2c_act import A2C

# Environment.
flags.DEFINE_integer('max_steps', 1000, 'steps for agent to try in the environment')
flags.DEFINE_integer('seed', 42, 'seed for random number generation')
flags.DEFINE_integer('n_acton', 9, 'total type of action to choose')

# Agent.
flags.DEFINE_integer('num_hidden_layers', 2, 'number of hidden layers')
flags.DEFINE_integer('num_units', 64, 'number of units per hidden layer')
flags.DEFINE_float('learning_rate', 1e-2, 'the learning rate')
flags.DEFINE_integer('sequence_length', 32, 'mumber of transitions to batch')
flags.DEFINE_float('td_lambda', 0.9, 'mixing parameter for boostrapping')
flags.DEFINE_float('discount', .99, 'discounting on the agent side')
flags.DEFINE_boolean('default', False, 'whether to use default agent')

FLAGS = flags.FLAGS
def run():
    print('Runing experiment in a2c(output the action) by pytorch...')
    env = World(FLAGS.max_steps, FLAGS.discount, FLAGS.seed, FLAGS.n_action)
    agent = A2C(
        obs_spec=env.observation_spec(),
        action_spec=env.action_spec(),
        max_sequence_length=FLAGS.sequence_length,
        network=
    )

if __name__ == '__main__':
    app.run(run)