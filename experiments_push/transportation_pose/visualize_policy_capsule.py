# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.b2PushWorld.TransportationPoseWorld import TransportationWorldConfig
from research_envs.envs.transportation_pose_capsule_env import TransportationEnvConfig, TransportationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

import cv2
import numpy as np
from stable_baselines3 import SAC

def _gen_rectangle_from_center_line(start_p, end_p, w):
    # Creates a rectangle from the center line (start_p, end_p)
    # with width = w
    # Returns a list of 4 points in counter-clockwise order
    line_angle = np.arctan2(end_p[1] - start_p[1], end_p[0] - start_p[0])
    vec = w * np.array([np.cos(line_angle + np.pi/2), np.sin(line_angle + np.pi/2)])
    return [
        (start_p[0] + vec[0], start_p[1] + vec[1]),
        (start_p[0] - vec[0], start_p[1] - vec[1]),
        (end_p[0] - vec[0], end_p[1] - vec[1]),
        (end_p[0] + vec[0], end_p[1] + vec[1])
    ]

def render():
    # pass
    screen = env.render()

    self = env.cur_env

    # Draw corridor
    w = self.max_corr_width
    min_sg = (self.world.goal['pos'].x, self.world.goal['pos'].y)
    if min_sg is not None:
        corridor = _gen_rectangle_from_center_line((self.start_obj_pos.x, self.start_obj_pos.y), min_sg, w)
        corridor = [self.world.worldToScreen(v) for v in corridor]
        cv2.polylines(screen, [np.array(corridor)], isClosed=True, color=(0, 255, 0), thickness=4)
        sg_radius = int(w * self.world.pixels_per_meter)
        cv2.circle(screen, self.world.worldToScreen(min_sg), sg_radius, (0, 255, 0), thickness=4)
        cv2.circle(
            screen, self.world.worldToScreen((self.start_obj_pos.x, self.start_obj_pos.y)), 
            sg_radius, (0, 255, 0), thickness=4)

    scene_buffer.PushFrame(screen)
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
    max_corr_width=10.0
)

obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        'empty',
        # 'empty', 'frame', 'horizontal_corridor', 'vertical_corridor', '4_circles_wide',
        # '1_circle', '1_rectangle', '1_triangle',
        # 'circle_line', 'small_4_circles',
        # '4_circles', 'sparse_1', 'sparse_2',
    ]
}
env = TransportationMixEnv(config, obs_l_dict)

model = SAC.load('model_ckp/progress_sac_rectangle_tolerance_pi18_pos_tol_2_reward_scale_10_capsule_width_10')
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

