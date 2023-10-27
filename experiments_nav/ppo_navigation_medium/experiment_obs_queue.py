# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.envs.navigation_env_obs import NavigationEnvConfig, NavigationMixEnv
from research_envs.b2PushWorld.NavigationWorld import NavigationWorldConfig
from research_envs.envs.obstacle_repo import obstacle_l_dict
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import torch

import os

config = NavigationEnvConfig(
    world_config= NavigationWorldConfig(
        obstacle_l = [],
        n_rays = 24,
        range_max = 25.0,
        agent_force_length=1.0
    ),
    max_steps = 300,
    previous_obs_queue_len = 5
)
obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        'empty', 'frame', 'horizontal_corridor', 'vertical_corridor', '4_circles_wide',
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

obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        'empty', 'frame', 'horizontal_corridor', 'vertical_corridor', '4_circles_wide',
        '1_circle', '1_rectangle', '1_triangle',
        'circle_line', 'small_4_circles',
        '4_circles', 'sparse_1', 'sparse_2',
        'corridor', 'crooked_corridor',
        '16_circles', '25_circles', '49_circles',
    ]
}
eval_env = NavigationMixEnv(config, obs_l_dict)

policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[512, 512, 512], vf=[512, 512, 512]))

model = PPO(
    "MlpPolicy", env,
    ent_coef=0.01,
    policy_kwargs=policy_kwargs,
    verbose=1, tensorboard_log="./tensorboard_dir/")
print(model.policy)

# Create dir 
ckp_dir = 'model_ckp'
if not os.path.exists(ckp_dir): 
    os.makedirs(ckp_dir) 

# Create log dir where evaluation results will be saved
eval_log_dir = "./eval_logs/"
os.makedirs(eval_log_dir, exist_ok=True)

eval_callback = EvalCallback(eval_env, best_model_save_path=eval_log_dir,
                              log_path=eval_log_dir, eval_freq=50000,
                              n_eval_episodes=400, deterministic=True,
                              render=False)


name = 'ppo_no_progress_3_512_obs_queue_5'

model.learn(
    total_timesteps=5000000, log_interval=10, progress_bar=True, reset_num_timesteps=True,
    callback=eval_callback,
    tb_log_name=name)
model.save(os.path.join(ckp_dir, "model_ckp_0"))
# for i in range(1, 20):
#     model.learn(
#         total_timesteps=250000, log_interval=10, progress_bar=True, reset_num_timesteps=False,
#         callback=eval_callback,
#         tb_log_name=name)
#     model.save(os.path.join(ckp_dir, "model_ckp_"+str(i)))