import numpy as np
import cv2
from . import visualizer


def rollout(env, agent, rollout_steps, hooks=[]):
    obs = env.reset()
    for step in range(rollout_steps):
        action = agent.act(obs)
        result = env.step(action)
        obs, rew, done, info = result
        for hook in hooks:
            hook(env, agent, result, step)
        if done:
            break
        env.render()
