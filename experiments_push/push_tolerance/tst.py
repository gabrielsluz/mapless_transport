# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.b2PushWorld.TransportationPoseWorld import TransportationWorldConfig
from research_envs.envs.transportation_pose_capsule_conditioned_env import TransportationEnvConfig, TransportationEnv
from research_envs.envs.object_repo import object_desc_dict

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback

import gymnasium as gym
from gymnasium.spaces import Dict, Box

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
exp_name = 'obj_{}_pos_tol_{}_angle_tol{}_caps_{}_{}_batch_sz_256_her_4'.format(
    obj_id, 
    int(100*pos_tol),
    int(100*angle_tol),
    caps_start,
    caps_end
)
angle_tol = np.deg2rad(angle_tol)

# Environment Wrapper that simply puts observation ina dict {'obs': observation}
# and returns the dict as observation
class DictWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super().__init__(env)
        self.observation_space = Dict({
            'observation':env.observation_space,
            "desired_goal": Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "achieved_goal": Box(low=0, high=1, shape=(2,), dtype=np.float32),
            })

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        obs, info = self.env.reset(**kwargs)

        return {'obs':obs}, info

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict) observation, reward, is this a final state (episode finished),
        is the max number of steps reached (episode finished artificially), additional informations
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return {'obs':obs}, reward, terminated, truncated, info

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
env = DictWrapper(TransportationEnv(config))

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
eval_env = DictWrapper(TransportationEnv(eval_env_config))

###################### POLICY ######################
policy_kwargs = dict(net_arch=[256, 256, 256])
model = SAC(
    "MultiInputPolicy", env,
    policy_kwargs=policy_kwargs,
    batch_size=256,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ),
    verbose=1, tensorboard_log="./tensorboard_dir/")
print(model.policy)

# print(model.env)

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
