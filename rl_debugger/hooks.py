from abc import abstractmethod
from abc import ABCMeta

class StepHook(metaclass=ABCMeta):
    """Hook function that will be called in rollout.
    Any callable that accepts (env, agent, result, step) as arguments can be used as a hook.
    """
    
    @abstractmethod
    def __call__(self, env, agent, result, step):
        """Call the hook.
        Args:
            env: Environment.
            agent: RL agent
            result: obs, rew, done, info.
            step: Current timestep.
        """
        raise NotImplementedError