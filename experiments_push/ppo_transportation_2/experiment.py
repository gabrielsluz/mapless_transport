# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.b2PushWorld.TransportationWorld import TransportationWorldConfig
from research_envs.envs.transportation_env import TransportationEnvConfig, TransportationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from stable_baselines3 import PPO
import torch

import os

config = TransportationEnvConfig(
    world_config= TransportationWorldConfig(
        obstacle_l = [],
        object_l=[{'name':'Circle', 'radius':4.0}],
        n_rays = 24,
        range_max = 25.0,
        force_length=1.0
    ),
    max_steps = 300,
    previous_obs_queue_len = 0
)

# Create dir 
ckp_dir = 'model_ckp'
if not os.path.exists(ckp_dir): 
    os.makedirs(ckp_dir) 

obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        'empty','frame', 'horizontal_corridor', 'vertical_corridor','4_circles_wide',
        '1_circle', '1_rectangle', '1_triangle'
    ]
}
env = TransportationMixEnv(config, obs_l_dict)

# policy_kwargs = dict(activation_fn=torch.nn.ReLU,
#                      net_arch=dict(pi=[64, 32, 32], vf=[64, 32, 32]))
model = PPO(
    "MlpPolicy", env,
    ent_coef=0.01,
    # policy_kwargs=policy_kwargs,
    verbose=1, tensorboard_log="./tensorboard_dir/")

name = 'ppo_original_success_20'

model.learn(
    total_timesteps=250000, log_interval=10, progress_bar=True, reset_num_timesteps=True,
    tb_log_name=name)
model.save(os.path.join(ckp_dir, name))

for i in range(1, 20):
    model.learn(
        total_timesteps=250000, log_interval=10, progress_bar=True, reset_num_timesteps=False,
        tb_log_name=name
        )
    model.save(os.path.join(ckp_dir, name))


