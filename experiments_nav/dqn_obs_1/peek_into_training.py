# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.envs.navigation_env import NavigationEnvConfig, NavigationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from research_envs.b2PushWorld.NavigationWorld import NavigationWorldConfig
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

import cv2
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

def key_to_action(key):
    action = -1
    if key == 113: #q
        action = 5
    elif key == 119: # w
        action = 6
    elif key == 101: # e
        action = 7
    elif key == 100: # d
        action = 0
    elif key == 99: # c
        action = 1
    elif key == 120: # x
        action = 2
    elif key == 122: # z
        action = 3
    elif key == 97: # a
        action = 4
    return action

# action_to_nl, but with lr (lower right), ur (upper right), r (right) ...
action_to_nl = {
    0: 'right      ',
    1: 'lower right',
    2: 'down       ',
    3: 'lower left ',
    4: 'left       ',
    5: 'upper left ',
    6: 'up         ',
    7: 'upper right'
}



def render():
    scene_buffer.PushFrame(env.render())
    scene_buffer.Draw()
    cv2.waitKey(1)

config = NavigationEnvConfig(
    world_config= NavigationWorldConfig(
        obstacle_l = [],
        n_rays = 8,
        range_max = 25.0
    ),
    max_steps = 200,
    previous_obs_queue_len = 2
)
obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        'circle_line',
        #'circle_line', 'small_4_circles', 'empty'
        #'small_4_circles', '16_circles', '25_circles', '49_circles',
        #'empty', 'circle_line', 'small_4_circles',
        #'1_circle', '1_rectangle', '1_triangle', 
        #'4_circles', '16_circles', 'corridor', 'crooked_corridor',
        #'sparse_1', 'sparse_2'
    ]
    #['4_circles', 'corridor', 'crooked_corridor', '16_circles', 'sparse_1', 'sparse_2']
}
env = NavigationMixEnv(config, obs_l_dict)
print(check_env(env))

model = PPO.load("model_ckp_0", env=env)
print(model.policy)

# Rollouts:
model.rollout_buffer.reset()
print('Full = ', model.rollout_buffer.full)
total_timesteps, dummy_callback = callback = model._setup_learn(
    2000,
    None,
    True,
    'Peak_into_training',
    False,
)
model.collect_rollouts(model.env, dummy_callback, model.rollout_buffer, n_rollout_steps=model.n_steps)
print('Full = ', model.rollout_buffer.full)
aux = model.rollout_buffer.get(1)
for o in aux:
    print(o)
    break
# print(model.rollout_buffer.get(5)[0])

# End rollouts

scene_buffer = CvDrawBuffer(window_name="Simulation", resolution=(1024,1024))

render()
obs, info = env.reset()
acc_reward = 0
while True:
    # Input handling - requires a cv2 window running => env.render()
    dt = 1.0 / 60.0 #1.0 / 60.0
    key = 0xFF & cv2.waitKey(int(dt * 1000.0)) # Sets default key = 255
    if key == 27: break # Esc key
    action = key_to_action(key)
    if action != -1:
        #action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        acc_reward += reward
        render()
        if terminated or truncated:
            obs, info = env.reset()
            print('Reward: ', acc_reward)
            acc_reward = 0
            render()
        
        obs_b = torch.from_numpy(np.expand_dims(obs, axis=0))
        print('Reward: ', round(reward, 2))
        print('Action probabilities:')
        act_dist = model.policy.get_distribution(obs_b)
        for i in range(8):
            p = np.exp(act_dist.log_prob(torch.Tensor([i])).item())
            v = model.policy.evaluate_actions(obs_b, torch.Tensor([i]))[0].item()
            print('A: ', action_to_nl[i], ' P: ', round(p, 2))
        print('Value of best: ', round(model.policy.predict_values(obs_b).item(), 2))
        print()

