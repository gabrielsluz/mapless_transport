# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.b2PushWorld.TransportationWorld import TransportationWorldConfig
from research_envs.envs.transportation_env import TransportationEnvConfig, TransportationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from stable_baselines3 import PPO

import os

config = TransportationEnvConfig(
    world_config= TransportationWorldConfig(
        obstacle_l = [],
        object_l=[{'name':'Circle', 'radius':4.0}],
        n_rays = 24,
        range_max = 25.0,
        force_length=2.0
    ),
    max_steps = 300,
    previous_obs_queue_len = 0
)

def run_experiment(map_name):
    obs_l_dict = {
        k: obstacle_l_dict[k] 
        for k in [
            map_name
        ]
    }
    env = TransportationMixEnv(config, obs_l_dict)

    model = PPO(
        "MlpPolicy", env,
        ent_coef=0.01,
        verbose=1, tensorboard_log="./tensorboard_dir/")
    
    model.learn(
        total_timesteps=200000, log_interval=10, progress_bar=True, reset_num_timesteps=True,
        tb_log_name=map_name)
    model.save(os.path.join(ckp_dir, map_name))

# Create dir 
ckp_dir = 'model_ckp'
if not os.path.exists(ckp_dir): 
    os.makedirs(ckp_dir) 

for m_n in [
    'empty', 'frame', 'horizontal_corridor', 'vertical_corridor','4_circles_wide',
    '1_circle', '1_rectangle', '1_triangle'
    ]:
    run_experiment(m_n)