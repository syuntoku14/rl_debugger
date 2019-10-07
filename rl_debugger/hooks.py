from abc import abstractmethod
from abc import ABCMeta
import cv2
import numpy as np
from . import visualizer
from .wrappers import SaliencyWrapper


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

class VideoHook(StepHook):
    def __init__(self, video_writer):
        self.video_writer = video_writer

    def __call__(self, env, agent, result, step):
        # Red: value, Blue: policy
        obs, _, _, _ = result
        map_img, sight_img = env.render(return_rgb=True)
        map_img = cv2.cvtColor(map_img, cv2.COLOR_RGB2BGR)
        self.video_writer.write(map_img)

    def close_video_writer(self):
        self.video_writer.release()


class SaliencyHook(StepHook):
    def __init__(self, model, phi, video_writer=None):
        self.video_writer = video_writer
        self.phi = phi
        self.model = SaliencyWrapper(model, phi)

    def __call__(self, env, agent, result, step):
        # Red: value, Blue: policy
        obs, _, _, _ = result
        map_img, sight_img = env.render(return_rgb=True)
        p_map, v_map = visualizer.get_saliency_map(self.model, obs)
        result_img = visualizer.plot_saliency_on_img(p_map, sight_img, channel=2)
        result_img = visualizer.plot_saliency_on_img(v_map, result_img, channel=0)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        map_img = cv2.cvtColor(map_img, cv2.COLOR_RGB2BGR)
        result_img = np.concatenate((map_img, result_img), axis=1)
        if self.video_writer is not None:
            self.video_writer.write(result_img)
        cv2.imshow('saliency', result_img)
        cv2.waitKey(10)

    def close_video_writer(self):
        self.video_writer.release()
