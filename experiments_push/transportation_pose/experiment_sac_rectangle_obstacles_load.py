# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.b2PushWorld.TransportationPoseWorld import TransportationWorldConfig
from research_envs.envs.transportation_pose_env import TransportationEnvConfig, TransportationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from stable_baselines3 import SAC

import os
import json
import numpy as np


object_desc = {'name': 'Rectangle', 'height': 10.0, 'width': 5.0}
print(object_desc)

config = TransportationEnvConfig(
    world_config= TransportationWorldConfig(
        obstacle_l = [],
        object_l=[
            object_desc
            ],
        n_rays = 24,
        range_max = 25.0,
        agent_type = 'continuous',
        max_force_length=5.0,
        min_force_length=0.1,
        goal_tolerance={'pos':2, 'angle':np.pi/18}
    ),
    max_steps = 500,
    previous_obs_queue_len = 2,
    reward_scale=10.0
)

exp_name = 'progress_sac_rectangle_tolerance_pi18_pos_tol_2_reward_scale_10_map_empty_load_then_middle'

obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        'empty', '1_circle', '1_rectangle', '1_triangle'
    ]
}
env = TransportationMixEnv(config, obs_l_dict)

# model = SAC(
#     "MlpPolicy", env,
#     verbose=1, tensorboard_log="./tensorboard_dir/")
model = SAC.load('model_ckp/progress_sac_rectangle_tolerance_pi18_pos_tol_2_reward_scale_10_map_empty')
print(model.policy)

model.set_env(env, force_reset=True)

# Create dir 
ckp_dir = 'model_ckp'
if not os.path.exists(ckp_dir): 
    os.makedirs(ckp_dir) 

for _ in range(50):
    model.learn(
        total_timesteps=50000, log_interval=10, progress_bar=True, reset_num_timesteps=False,
        tb_log_name=exp_name)
    model.save(os.path.join(ckp_dir, exp_name))