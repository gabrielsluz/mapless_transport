# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.envs.navigation_env import NavigationEnvConfig, NavigationEnv
from research_envs.b2PushWorld.NavigationWorld import NavigationWorldConfig
from stable_baselines3 import DQN

env = NavigationEnv(
    NavigationEnvConfig(
        max_steps=200,
        world_config=NavigationWorldConfig(
            obstacle_l = [],
            n_rays = 16,
            range_max = 4.0
        )))

model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_dir/")
model.learn(total_timesteps=100000, log_interval=100, progress_bar=True)
model.save("model_ckp")
# import gymnasium as gym

# from stable_baselines3 import DQN

# env = gym.make("CartPole-v1", render_mode=None)

# model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_cartpole_tensorboard/")
# model.learn(total_timesteps=100000, log_interval=1, progress_bar=True)
# model.save("dqn_cartpole")

# print('Done training')
# del model # remove to demonstrate saving and loading

# model = DQN.load("dqn_cartpole")

# env = gym.make("CartPole-v1", render_mode="human")
# obs, info = env.reset()
# acc_reward = 0
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     acc_reward += reward
#     if terminated or truncated:
#         obs, info = env.reset()
#         print('Reward: ', acc_reward)
#         acc_reward = 0