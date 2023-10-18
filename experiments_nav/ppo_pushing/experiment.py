# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.envs.pushing_pose_env import PushingEnvConfig, PushingEnv
from research_envs.b2PushWorld.PushSimulatorPose import PushSimulatorConfig
from research_envs.envs.rewards import RewardFunctions
from stable_baselines3 import PPO

import os

config = PushingEnvConfig(
    reward_fn_id=RewardFunctions.PROGRESS,
    max_steps=200,
    terminate_obj_dist = 14.0,
    push_simulator_config=PushSimulatorConfig(
        obj_proximity_radius=14.0,
        objTuple = (
            {'name':'Circle', 'radius':4.0},
        )
    ),
)
env = PushingEnv(config=config)


model = PPO(
    "MlpPolicy", env,
    ent_coef=0.01,
    verbose=1, tensorboard_log="./tensorboard_dir/")

# Create dir 
ckp_dir = 'model_ckp'
if not os.path.exists(ckp_dir): 
    os.makedirs(ckp_dir) 

model.learn(total_timesteps=250000, log_interval=10, progress_bar=True, reset_num_timesteps=True)
model.save(os.path.join(ckp_dir, "model_ckp_0"))
for i in range(1, 20):
    model.learn(total_timesteps=250000, log_interval=10, progress_bar=True, reset_num_timesteps=False)
    model.save(os.path.join(ckp_dir, "model_ckp_"+str(i)))