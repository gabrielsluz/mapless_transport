# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.b2PushWorld.TransportationPoseWorld import TransportationWorldConfig
from research_envs.envs.transportation_pose_env import TransportationEnvConfig, TransportationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

import cv2
import numpy as np
from stable_baselines3 import SAC

def render():
    # pass
    scene_buffer.PushFrame(env.render())
    scene_buffer.Draw()
    cv2.waitKey(100)

object_desc = {'name': 'Rectangle', 'height': 10.0, 'width': 5.0}
print(object_desc)

config = TransportationEnvConfig(
    world_config= TransportationWorldConfig(
        obstacle_l = [],
        object_l=[
            object_desc
            ],
        n_rays = 24,
        range_max = 25.0,
        agent_type = 'continuous',
        max_force_length=5.0,
        min_force_length=0.1,
        goal_tolerance={'pos':2, 'angle':np.pi/18}
    ),
    max_steps = 100,
    previous_obs_queue_len = 2,
    reward_scale=10.0
)

obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        'empty', '1_circle', '1_rectangle', '1_triangle'
        # 'empty', 'frame', 'horizontal_corridor', 'vertical_corridor', '4_circles_wide',
        # '1_circle', '1_rectangle', '1_triangle',
        # 'circle_line', 'small_4_circles',
        # '4_circles', 'sparse_1', 'sparse_2',
    ]
}
env = TransportationMixEnv(config, obs_l_dict)

model = SAC.load('model_ckp/progress_sac_rectangle_tolerance_pi18_pos_tol_2_reward_scale_10_map_large_obstacle_middle')
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

