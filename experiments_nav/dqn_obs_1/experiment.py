# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.envs.navigation_env import NavigationEnvConfig, NavigationMixEnv
from research_envs.b2PushWorld.NavigationWorld import NavigationWorldConfig
from research_envs.envs.obstacle_repo import obstacle_l_dict
from stable_baselines3 import PPO

config = NavigationEnvConfig(
    world_config= NavigationWorldConfig(
        obstacle_l = [],
        n_rays = 24,
        range_max = 8.0
    ),
    max_steps=200
)
obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in ['sparse_1', '4_circles']
}
env = NavigationMixEnv(config, obs_l_dict)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_dir/")
model.learn(total_timesteps=250000, log_interval=10, progress_bar=True)
model.save("model_ckp")