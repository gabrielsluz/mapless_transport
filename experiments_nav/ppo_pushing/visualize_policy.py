# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.envs.pushing_pose_env import PushingEnvConfig, PushingEnv
from research_envs.b2PushWorld.PushSimulatorPose import PushSimulatorConfig
from research_envs.envs.rewards import RewardFunctions
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

import cv2
from stable_baselines3 import PPO

# def render():
#     # pass
#     scene_buffer.PushFrame(env.render())
#     scene_buffer.Draw()
#     cv2.waitKey(1)

config = PushingEnvConfig(
    reward_fn_id=RewardFunctions.PROGRESS,
    max_steps=200,
    terminate_obj_dist = 14.0,
    push_simulator_config=PushSimulatorConfig(
        obj_proximity_radius=14.0,
        objTuple = (
            {'name':'Circle', 'radius':4.0},
        )
    ),
)
env = PushingEnv(config=config)

model = PPO.load("model_ckp/model_ckp_1")
print(model.policy)

# scene_buffer = CvDrawBuffer(window_name="Simulation", resolution=(1024,1024))

env.render()
obs, info = env.reset()
acc_reward = 0
success_l = []
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    acc_reward += reward
    env.render()
    if terminated or truncated:
        success_l.append(info['is_success'])
        obs, info = env.reset()
        print('Reward: ', acc_reward)
        acc_reward = 0
        print('Success rate: ', sum(success_l) / len(success_l))

