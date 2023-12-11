# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.b2PushWorld.TransportationWorld import TransportationWorldConfig
from research_envs.envs.transportation_env import TransportationEnvConfig, TransportationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from stable_baselines3 import SAC, PPO

import os


# Create SAC agent and print policy
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

obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        'empty'
    ]
}
env = TransportationMixEnv(config, obs_l_dict)


# Create a MLP Policy for SAC with 2 hidden layers of size 64
policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

# model = SAC(
#     'MlpPolicy', env,
#     verbose=1, policy_kwargs=policy_kwargs)
model = PPO(
    'MlpPolicy', env,
    verbose=1, policy_kwargs=policy_kwargs)
print(model.policy)