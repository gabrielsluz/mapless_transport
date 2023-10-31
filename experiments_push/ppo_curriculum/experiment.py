# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.b2PushWorld.TransportationWorld import TransportationWorldConfig
from research_envs.envs.transportation_env import TransportationEnvConfig, TransportationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
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
        'empty', 'frame', 'horizontal_corridor', 'vertical_corridor','4_circles_wide',
        '1_circle', '1_rectangle', '1_triangle'
    ]
}
env = TransportationMixEnv(config, obs_l_dict)

policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[64, 64, 64], vf=[64, 64, 64]))

model = PPO(
    "MlpPolicy", env,
    ent_coef=0.01,
    policy_kwargs=policy_kwargs,
    verbose=1, tensorboard_log="./tensorboard_dir/")

# Create log dir where evaluation results will be saved
eval_log_dir = "./eval_logs/"
os.makedirs(eval_log_dir, exist_ok=True)

eval_callback = EvalCallback(env, best_model_save_path=eval_log_dir,
                              log_path=eval_log_dir, eval_freq=50000,
                              n_eval_episodes=400, deterministic=True,
                              render=False)


name = 'ppo_base'

model.learn(
    total_timesteps=3000000, log_interval=10, progress_bar=True, reset_num_timesteps=True,
    callback=eval_callback,
    tb_log_name=name)
model.save(os.path.join(ckp_dir, name))


