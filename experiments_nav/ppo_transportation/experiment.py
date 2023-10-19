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
        range_max = 5.0,
        force_length=2.0
    ),
    max_steps = 300,
    previous_obs_queue_len = 0
)
obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        'empty'
    ]
}
env = TransportationMixEnv(config, obs_l_dict)

model = PPO(
    "MlpPolicy", env,
    ent_coef=0.01,
    # learning_rate=0.0001,
    # n_steps=8192,
    # n_epochs=5,
    # batch_size=256,
    verbose=1, tensorboard_log="./tensorboard_dir/")

# Create dir 
ckp_dir = 'model_ckp'
if not os.path.exists(ckp_dir): 
    os.makedirs(ckp_dir) 

model.learn(total_timesteps=200000, log_interval=10, progress_bar=True, reset_num_timesteps=True)
model.save(os.path.join(ckp_dir, "model_ckp_0"))
for i in range(1, 20):
    model.learn(total_timesteps=200000, log_interval=10, progress_bar=True, reset_num_timesteps=False)
    model.save(os.path.join(ckp_dir, "model_ckp_"+str(i)))