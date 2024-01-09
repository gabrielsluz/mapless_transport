# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.envs.navigation_env import NavigationEnvConfig, NavigationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from research_envs.b2PushWorld.NavigationWorld import NavigationWorldConfig
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

import cv2
from stable_baselines3 import SAC

def render():
    # pass
    scene_buffer.PushFrame(env.render())
    scene_buffer.Draw()
    cv2.waitKey(100)

config = NavigationEnvConfig(
    world_config= NavigationWorldConfig(
        obstacle_l = [],
        n_rays = 24,
        range_max = 25.0,
        agent_type = 'continuous',
        max_force_length=5.0,
        min_force_length=0.1
    ),
    max_steps = 300,
    previous_obs_queue_len = 2,
    reward_scale=10.0
)

obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        'empty', 'frame', 'horizontal_corridor', 'vertical_corridor', '4_circles_wide',
        '1_circle', '1_rectangle', '1_triangle',
        'circle_line', 'small_4_circles',
        '4_circles', 'sparse_1', 'sparse_2',
    ]
}
env = NavigationMixEnv(config, obs_l_dict)

model = SAC.load('model_ckp/progress_sac_medium_reward_scale_10_prev_obs_2')
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

