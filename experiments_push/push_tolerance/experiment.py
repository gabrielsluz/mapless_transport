# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.b2PushWorld.TransportationPoseWorld import TransportationWorldConfig
from research_envs.envs.transportation_pose_capsule_conditioned_env import TransportationEnvConfig, TransportationEnv
from research_envs.envs.object_repo import object_desc_dict

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

import os
import numpy as np

"""
Trains an RL model.
Output:
    - model_ckp/obj_{obj_id}.zip : last trained model
    - best_model_ckp/obj_{obj_id}.zip : best model according to evaluation loop
    - tensorboard_dir/obj_{obj_id} : directory containing the tensorboard logs
"""

###################### OBJECT ID ######################
obj_id = int(sys.argv[1])
pos_tol = float(sys.argv[2])
angle_tol = float(sys.argv[3])
caps_start = int(sys.argv[4])
caps_end = int(sys.argv[5])
exp_name = 'obj_{}_pos_tol_{}_angle_tol{}_caps_{}_{}_batch_sz_256'.format(
    obj_id, 
    int(100*pos_tol),
    int(100*angle_tol),
    caps_start,
    caps_end
)
angle_tol = np.deg2rad(angle_tol)

###################### TRAINING ENVIRONMENT ######################
config = TransportationEnvConfig(
    world_config= TransportationWorldConfig(
        obstacle_l = [],
        object_l=[object_desc_dict[obj_id]],
        n_rays = 0,
        agent_type = 'continuous',
        max_force_length=5.0,
        min_force_length=0.1,
        goal_tolerance={'pos':pos_tol, 'angle':angle_tol},
        max_obj_dist=8.0
    ),
    max_steps = 500,
    previous_obs_queue_len = 0,
    reward_scale=10.0,
    corridor_width_range = (float(caps_start), float(caps_end))
)
env = TransportationEnv(config)

###################### EVALUATION ENVIRONMENT ######################
eval_env_config = TransportationEnvConfig(
    world_config= TransportationWorldConfig(
        obstacle_l = [],
        object_l=[object_desc_dict[obj_id]],
        n_rays = 0,
        agent_type = 'continuous',
        max_force_length=5.0,
        min_force_length=0.1,
        goal_tolerance={'pos':pos_tol, 'angle':angle_tol},
        max_obj_dist=8.0
    ),
    max_steps = 200,
    previous_obs_queue_len = 0,
    reward_scale=10.0,
    corridor_width_range = (float(caps_start), float(caps_end))
)
eval_env = TransportationEnv(eval_env_config)

###################### POLICY ######################
policy_kwargs = dict(net_arch=[256, 256, 256])
model = SAC(
    "MlpPolicy", 
    env,
    policy_kwargs=policy_kwargs,
    batch_size=256,
    verbose=1, tensorboard_log="./tensorboard_dir/")
print(model.policy)

###################### TRAINING ######################
# Create dir 
ckp_dir = 'model_ckp'
if not os.path.exists(ckp_dir): 
    os.makedirs(ckp_dir) 

eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path=ckp_dir+'/'+exp_name,
    log_path=ckp_dir+'/'+exp_name,
    n_eval_episodes=1_000,
    eval_freq=100_000,
    deterministic=True, render=False)

# Main training loop
model.learn(
    total_timesteps=4_000_000, log_interval=100, progress_bar=True, reset_num_timesteps=True,
    callback=eval_callback,
    tb_log_name=exp_name)


# Train and eval more frequently
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path=ckp_dir+'/'+exp_name,
    log_path=ckp_dir+'/'+exp_name,
    n_eval_episodes=2_000,
    eval_freq=5_000,
    deterministic=True, render=False)

model.learn(
    total_timesteps=1_000_000, log_interval=100, progress_bar=True, reset_num_timesteps=False,
    callback=eval_callback,
    tb_log_name=exp_name)
