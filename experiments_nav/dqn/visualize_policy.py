# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.envs.navigation_env import NavigationEnvConfig, NavigationEnv
from research_envs.b2PushWorld.NavigationWorld import NavigationWorldConfig
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

import cv2
from stable_baselines3 import DQN

def render():
    scene_buffer.PushFrame(env.render())
    scene_buffer.Draw()
    cv2.waitKey(1)

env = NavigationEnv(
    NavigationEnvConfig(
        max_steps=200,
        world_config=NavigationWorldConfig(
            obstacle_l = [],
            n_rays = 16,
            range_max = 4.0
        )))

model = DQN.load("model_ckp")
scene_buffer = CvDrawBuffer(window_name="Simulation", resolution=(1024,1024))

render()
obs, info = env.reset()
acc_reward = 0
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    acc_reward += reward
    render()
    if terminated or truncated:
        obs, info = env.reset()
        print('Reward: ', acc_reward)
        acc_reward = 0

