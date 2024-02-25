# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.b2PushWorld.TransportationPoseWorld import TransportationWorldConfig
from research_envs.envs.transportation_pose_capsule_conditioned_env import TransportationEnvConfig, TransportationEnv
from research_envs.envs.object_repo import object_desc_dict
from stable_baselines3 import SAC

import os
import numpy as np
import time

"""
Trains an RL model.
Output:
    - model_ckp/obj_{obj_id}.zip : last trained model
    - best_model_ckp/obj_{obj_id}.zip : best model according to evaluation loop
    - tensorboard_dir/obj_{obj_id} : directory containing the tensorboard logs
"""

# Function for evaluation the policy and computing the success rate for n_episodes
def evaluate_policy(model, eval_env, n_episodes):
    success = 0
    for _ in range(n_episodes):
        obs = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            if done:
                success += info['is_success']
    return success / n_episodes

###################### OBJECT ID ######################
obj_id = int(sys.argv[1])
exp_name = 'obj_' + str(obj_id)

###################### TRAINING ENVIRONMENT ######################
config = TransportationEnvConfig(
    world_config= TransportationWorldConfig(
        obstacle_l = [],
        object_l=[object_desc_dict[obj_id]],
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
        goal_tolerance={'pos':2, 'angle':np.pi/18},
        max_obj_dist=10.0
    ),
    max_steps = 100,
    previous_obs_queue_len = 0,
    reward_scale=10.0,
    corridor_width_range = (10.0, 20.0)
)
eval_env = TransportationEnv(eval_env_config)

###################### POLICY ######################
policy_kwargs = dict(net_arch=[256, 256, 256])
model = SAC(
    "MlpPolicy", env,
    policy_kwargs=policy_kwargs,
    verbose=1, tensorboard_log="./tensorboard_dir/")
print(model.policy)


###################### TRAINING ######################
# Create dir 
ckp_dir = 'model_ckp'
if not os.path.exists(ckp_dir): 
    os.makedirs(ckp_dir) 

best_ckp_dir = 'best_model_ckp'
if not os.path.exists(best_ckp_dir): 
    os.makedirs(best_ckp_dir) 
best_score = None

model.learn(
    total_timesteps=1000, log_interval=10, progress_bar=True, reset_num_timesteps=True,
    tb_log_name=exp_name)
model.save(os.path.join(ckp_dir, exp_name))

for _ in range(500):
    model.learn(
        total_timesteps=1000, log_interval=10, progress_bar=True, reset_num_timesteps=False,
        tb_log_name=exp_name)
    model.save(os.path.join(ckp_dir, exp_name))

    # Evaluate the trained model
    start_time = time.time()
    score = evaluate_policy(model, eval_env, 100)
    end_time = time.time()
    print(f"The execution time of the line is: {end_time - start_time} seconds")

    print(f"Success rate: {score}")
    if best_score is None or score > best_score:
        best_score = score
        model.save(os.path.join(best_ckp_dir, exp_name))
