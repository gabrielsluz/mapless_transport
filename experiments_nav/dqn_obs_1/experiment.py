# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.envs.navigation_env import NavigationEnvConfig, NavigationEnv
from research_envs.b2PushWorld.NavigationWorld import NavigationWorldConfig
from stable_baselines3 import PPO

env = NavigationEnv(
    NavigationEnvConfig(
        max_steps=200,
        world_config=NavigationWorldConfig(
            obstacle_l = [
                {'name':'Circle', 'pos':(5.0, 5.0), 'radius':2.0},
                {'name':'Circle', 'pos':(10.0, 10.0), 'radius':5.0},
                {'name':'Circle', 'pos':(35.0, 35.0), 'radius':2.0},
                {'name':'Circle', 'pos':(45.0, 35.0), 'radius':2.0},
                {'name':'Circle', 'pos':(5.0, 35.0), 'radius':4.0},
                {'name':'Rectangle', 'pos':(25.0, 25.0), 'height':10.0, 'width':2.0}
            ],
            n_rays = 16,
            range_max = 4.0
        )))
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_dir/")
model.learn(total_timesteps=250000, log_interval=10, progress_bar=True)
model.save("model_ckp")