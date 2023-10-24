# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.envs.navigation_env import NavigationEnvConfig, NavigationMixEnv
from research_envs.b2PushWorld.NavigationWorld import NavigationWorldConfig
from research_envs.envs.obstacle_repo import obstacle_l_dict
from stable_baselines3 import PPO

import os

config = NavigationEnvConfig(
    world_config= NavigationWorldConfig(
        obstacle_l = [],
        n_rays = 32,
        range_max = 25.0,
        agent_type = 'forward',
        agent_width = 1.0,
        agent_height = 1.0,
        action_step_len = 10,
        action_velocity=4.0,
        action_l=[-3.0, -1.5, 0, 1.5, 3.0]
    ),
    max_steps = 300,
    previous_obs_queue_len = 0
)
obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        'empty', 'frame', 'horizontal_corridor', 'vertical_corridor','4_circles_wide',
        '1_circle', '1_rectangle', '1_triangle',
        'circle_line', 'small_4_circles',
        '4_circles', 'sparse_1', 'sparse_2',
        'corridor', 'crooked_corridor',
        '16_circles', '25_circles', '49_circles',
        # 'small_U', 'small_G',
        # 'U', 'G',
    ]
}
env = NavigationMixEnv(config, obs_l_dict)

model = PPO(
    "MlpPolicy", env,
    ent_coef=0.01,
    verbose=1, tensorboard_log="./tensorboard_dir/")

# Create dir 
ckp_dir = 'model_ckp'
if not os.path.exists(ckp_dir): 
    os.makedirs(ckp_dir) 

model.learn(
    total_timesteps=250000, log_interval=10, progress_bar=True, reset_num_timesteps=True,
    tb_log_name='ppo_forward_range_25_div_5_32_rays_minus_200_01_progress_medium_level_1_radius')
model.save(os.path.join(ckp_dir, "model_ckp_0"))
for i in range(1, 20):
    model.learn(
        total_timesteps=250000, log_interval=10, progress_bar=True, reset_num_timesteps=False,
        tb_log_name='ppo_forward_range_25_div_5_32_rays_minus_200_01_progress_medium_level_1_radius')
    model.save(os.path.join(ckp_dir, "model_ckp_"+str(i)))