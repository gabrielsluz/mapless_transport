# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.b2PushWorld.TransportationWorld import TransportationWorldConfig
from research_envs.envs.transportation_env import TransportationEnvConfig, TransportationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

import cv2
from stable_baselines3 import PPO

def render():
    # pass
    scene_buffer.PushFrame(env.render())
    scene_buffer.Draw()
    cv2.waitKey(1)

config = TransportationEnvConfig(
    world_config= TransportationWorldConfig(
        obstacle_l = [],
        n_rays = 32,
        range_max = 25.0,
        agent_type = 'forward',
        agent_width = 2.0,
        agent_height = 2.0,
        action_step_len = 10,
        action_velocity=4.0,
        action_l=[-3.0, -1.5, 0, 1.5, 3.0]
    ),
    max_steps=500,
    previous_obs_queue_len=0
)
obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        'empty', 
        # 'frame', 'horizontal_corridor', 'vertical_corridor','4_circles_wide',
        # '1_circle', '1_rectangle', '1_triangle'

        # 'empty', 'frame', 'horizontal_corridor', 'vertical_corridor','4_circles_wide',
        # '1_circle', '1_rectangle', '1_triangle',
        # 'circle_line', 'small_4_circles',
        # '4_circles', 'sparse_1', 'sparse_2',
        # 'corridor', 'crooked_corridor',
        # '16_circles', '25_circles', '49_circles',
        # 'sparse_3'
    ]
}
env = TransportationMixEnv(config, obs_l_dict)

model = PPO.load("model_ckp/model_ckp_4")
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

