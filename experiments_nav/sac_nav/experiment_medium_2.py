# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.envs.navigation_env import NavigationEnvConfig, NavigationMixEnv
from research_envs.b2PushWorld.NavigationWorld import NavigationWorldConfig
from research_envs.envs.obstacle_repo import obstacle_l_dict
from stable_baselines3 import SAC

import os
import json
import numpy as np

config = NavigationEnvConfig(
    world_config= NavigationWorldConfig(
        obstacle_l = [],
        n_rays = 72,
        range_max = 25.0,
        agent_type = 'continuous',
        max_force_length=5.0,
        min_force_length=0.1
    ),
    max_steps = 300,
    previous_obs_queue_len = 2,
    reward_scale=10.0
)

exp_name = 'progress_sac_medium_2_reward_scale_10_prev_obs_2_n_rays_72_layers_512_512'

obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        'empty', 'frame', 'horizontal_corridor', 'vertical_corridor', '4_circles_wide',
        '1_circle', '1_rectangle', '1_triangle',
        'circle_line', 'small_4_circles',
        '4_circles', 'sparse_1', 'sparse_2',
        '16_circles', '25_circles', '49_circles',
    ]
}
env = NavigationMixEnv(config, obs_l_dict)

policy_kwargs = dict(net_arch=dict(pi=[512, 512], qf=[512, 512]))

model = SAC(
    "MlpPolicy", env,
    policy_kwargs=policy_kwargs,
    verbose=1, tensorboard_log="./tensorboard_dir/")
print(model.policy)

# Create dir 
ckp_dir = 'model_ckp'
if not os.path.exists(ckp_dir): 
    os.makedirs(ckp_dir) 

model.learn(
    total_timesteps=25000, log_interval=10, progress_bar=True, reset_num_timesteps=True,
    tb_log_name=exp_name)
model.save(os.path.join(ckp_dir, exp_name))

for _ in range(50):
    model.learn(
        total_timesteps=25000, log_interval=10, progress_bar=True, reset_num_timesteps=False,
        tb_log_name=exp_name)
    model.save(os.path.join(ckp_dir, exp_name))


# for k in [
#         'empty', 'frame', 'horizontal_corridor', 'vertical_corridor', '4_circles_wide',
#         '1_circle', '1_rectangle', '1_triangle',
#         'circle_line', 'small_4_circles',
#         '4_circles', 'sparse_1', 'sparse_2',
#         'corridor', 'crooked_corridor',
#         '16_circles', '25_circles', '49_circles',
#         # 'small_U', 'small_G',
#         # 'U', 'G',
#     ]