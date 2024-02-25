# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.b2PushWorld.TransportationPoseWorld import TransportationWorldConfig
from research_envs.envs.transportation_pose_capsule_conditioned_env import TransportationEnvConfig, TransportationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from stable_baselines3 import SAC

import os
import json
import numpy as np


object_desc = {
    'name': 'MultiPolygons',
    'poly_vertices_l': [
        [[-0.03305392443269466, -2.6112598540991714],
         [-0.03305392443269466, 4.2639559668821585],
         [-2.6112598540991714, 4.2639559668821585],
         [-2.6112598540991714, -2.6112598540991714]],
         [[4.2639559668821585, -2.6112598540991714],
          [4.2639559668821585, -0.03305392443269466],
          [-0.03305392443269466, -0.03305392443269466],
          [-0.03305392443269466, -2.6112598540991714]]
  ]
}
print(object_desc)

config = TransportationEnvConfig(
    world_config= TransportationWorldConfig(
        obstacle_l = [],
        object_l=[
            object_desc
            ],
        n_rays = 0,
        agent_type = 'continuous',
        max_force_length=5.0,
        min_force_length=0.1,
        goal_tolerance={'pos':2, 'angle':np.pi/18},
        max_obj_dist=10.0
    ),
    max_steps = 500,
    previous_obs_queue_len = 0,
    reward_scale=10.0,
    corridor_width_range = (10.0, 20.0)
)

exp_name = 'L'

obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        'empty'
    ]
}
env = TransportationMixEnv(config, obs_l_dict)

policy_kwargs = dict(net_arch=[256, 256, 256])

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
    total_timesteps=50000, log_interval=10, progress_bar=True, reset_num_timesteps=True,
    tb_log_name=exp_name)
model.save(os.path.join(ckp_dir, exp_name))

for _ in range(500):
    model.learn(
        total_timesteps=50000, log_interval=10, progress_bar=True, reset_num_timesteps=False,
        tb_log_name=exp_name)
    model.save(os.path.join(ckp_dir, exp_name))
