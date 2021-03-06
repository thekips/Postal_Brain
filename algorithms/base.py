import abc
import dm_env

Action = int  # Only discrete-action agents for now.

class Agent(abc.ABC):
  """An agent consists of an action-selection mechanism and an update rule."""

  @abc.abstractmethod
  def select_action(self, timestep: dm_env.TimeStep) -> Action:
    """Takes in a timestep, samples from agent's policy, returns an action."""

  @abc.abstractmethod
  def update(
      self,
      timestep: dm_env.TimeStep,
      action: Action,
      new_timestep: dm_env.TimeStep,
  ) -> None:
    """Updates the agent given a transition."""