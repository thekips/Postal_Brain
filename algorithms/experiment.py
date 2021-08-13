from algorithms.base import Agent 
import dm_env
import torch

def run(
    agent: Agent,
    environment: dm_env.Environment,
    num_episodes: int,
    results_dir: str = 'models/actor_critic/default.pkl'
) -> None:
    '''
    Runs an agent on an enviroment.

    Args:
        agent: The agent to train and evaluate.
        environment: The environment to train on.
        num_episodes: Number of episodes to train for.
        verbose: Whether to also log to terminal.
    '''

    step = 1
    for episode in range(num_episodes):
        #Run an episode.
        timestep = environment.reset()

        while not timestep.last():
            action = agent.select_action(timestep)
            # print('The action is',action)
            new_timestep = environment.step(action)

            # Pass the (s, a, r, s')info to the agent.
            agent.update(timestep, action, new_timestep, step)

            # update timestep
            timestep = new_timestep
            step += 1
        
        # if(episode + 1) % 100 == 0:
        #     print("Episode %d success." % (episode + 1))
        print("Episode %d success." % (episode + 1))
        
        if True:
            torch.save(getattr(agent, '_network'), results_dir)

    print("We have finish it.")
