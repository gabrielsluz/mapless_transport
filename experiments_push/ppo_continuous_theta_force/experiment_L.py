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
        object_l=[{'name': 'MultiPolygons', 'poly_vertices_l':[[[0, 0], [0, 4], [12, 4], [12, 0]], [[0, 4], [0, 8], [4, 8], [4,4]]]}],
        n_rays = 0,
        agent_type = 'continuous',
        max_force_length=5.0,
        min_force_length=0.25
    ),
    max_steps = 500,
    previous_obs_queue_len = 0
)

exp_name = 'L_min_force_length_025_layers_256_256'

obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        'empty'
    ]
}
env = TransportationMixEnv(config, obs_l_dict)

policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
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

model.learn(
    total_timesteps=4000000, log_interval=10, progress_bar=True, reset_num_timesteps=True,
    tb_log_name=exp_name)
model.save(os.path.join(ckp_dir, exp_name))



# for m_n in [
#     'empty', 'frame', 'horizontal_corridor', 'vertical_corridor','4_circles_wide',
#     '1_circle', '1_rectangle', '1_triangle'
#     ]:
#     run_experiment(m_n)