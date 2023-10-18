# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.envs.navigation_env import NavigationEnvConfig, NavigationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from research_envs.b2PushWorld.NavigationWorld import NavigationWorldConfig
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

import cv2
from stable_baselines3 import PPO

def render():
    scene_buffer.PushFrame(env.render())
    scene_buffer.Draw()
    cv2.waitKey(1)

config = NavigationEnvConfig(
    world_config= NavigationWorldConfig(
        obstacle_l = [],
        n_rays = 24,
        range_max = 5.0
    ),
    max_steps = 200,
    previous_obs_queue_len = 5
)
obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        'circle_line', 'small_4_circles',
        '4_circles', 'sparse_1', 'sparse_2',
        '1_circle', '1_rectangle', '1_triangle',
        #'corridor', 'crooked_corridor',
        '16_circles', '25_circles', '49_circles',
        # '1_circle', '1_rectangle', '1_triangle',
        # 'corridor', 'crooked_corridor',
        # '16_circles', '25_circles', '49_circles',
        # 'small_U', 'small_G',
        # 'U', 'G'
        #'circle_line', 'small_4_circles', 'empty'
        #'small_4_circles', '16_circles', '25_circles', '49_circles',
        #'empty', 'circle_line', 'small_4_circles',
        #'1_circle', '1_rectangle', '1_triangle', 
        #'4_circles', '16_circles', 'corridor', 'crooked_corridor',
        #'sparse_1', 'sparse_2'
    ]
}
env = NavigationMixEnv(config, obs_l_dict)

model = PPO.load("model_ckp_1")
print(model.policy)

scene_buffer = CvDrawBuffer(window_name="Simulation", resolution=(1024,1024))

render()
obs, info = env.reset()
acc_reward = 0
success_l = []
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    acc_reward += reward
    render()
    if terminated or truncated:
        success_l.append(info['is_success'])
        obs, info = env.reset()
        print('Reward: ', acc_reward)
        acc_reward = 0
        print('Success rate: ', sum(success_l) / len(success_l))

