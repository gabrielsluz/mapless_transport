# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.b2PushWorld.TransportationPoseWorld import TransportationWorldConfig
from research_envs.envs.transportation_pose_capsule_conditioned_env import TransportationEnvConfig, TransportationMixEnv
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
    screen = env.render()
    start_pos = env.cur_env.start_obj_pos
    end_pos = env.cur_env.world.goal['pos']

    start = env.cur_env.world.worldToScreen((start_pos.x, start_pos.y))
    end = env.cur_env.world.worldToScreen((end_pos.x, end_pos.y))
    screen = cv2.line(screen, start, end, color=(0, 0, 255), thickness=2)

    # Draw a line ax + b = y
    corr_line = env.cur_env.corr_line
    a,b = corr_line[0], corr_line[1]

    # Find the paralel lines
    w = env.cur_env.corridor_width
    angle_line_to_x = np.arctan2(a, 1)

    start_pos = env.cur_env.start_obj_pos
    start_upper = (
        start_pos.x + w * np.cos(np.pi/2 + angle_line_to_x),
        start_pos.y + w * np.sin(np.pi/2 + angle_line_to_x)
    )
    start_lower = (
        start_pos.x + w * np.cos(-np.pi/2 + angle_line_to_x),
        start_pos.y + w * np.sin(-np.pi/2 + angle_line_to_x)
    )

    end_pos = env.cur_env.world.goal['pos']
    end_upper = (
        end_pos.x + w * np.cos(np.pi/2 + angle_line_to_x),
        end_pos.y + w * np.sin(np.pi/2 + angle_line_to_x)
    )
    end_lower = (
        end_pos.x + w * np.cos(-np.pi/2 + angle_line_to_x),
        end_pos.y + w * np.sin(-np.pi/2 + angle_line_to_x)
    )

    start = env.cur_env.world.worldToScreen(start_upper)
    end = env.cur_env.world.worldToScreen(end_upper)
    screen = cv2.line(screen, start, end, color=(0, 0, 255), thickness=2)

    start = env.cur_env.world.worldToScreen(start_lower)
    end = env.cur_env.world.worldToScreen(end_lower)
    screen = cv2.line(screen, start, end, color=(0, 0, 255), thickness=2)

    # Circles
    screen_pos = env.cur_env.world.worldToScreen(start_pos)
    cv2.circle(screen, screen_pos, int(w*env.cur_env.world.pixels_per_meter), color=(0, 0, 255), thickness=2)

    screen_pos = env.cur_env.world.worldToScreen(end_pos)
    cv2.circle(screen, screen_pos, int(w*env.cur_env.world.pixels_per_meter), color=(0, 0, 255), thickness=2)


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
        goal_tolerance={'pos':2, 'angle':np.pi},
        max_obj_dist=10.0
    ),
    max_steps = 500,
    previous_obs_queue_len = 0,
    reward_scale=10.0,
    corridor_width_range = (10.0, 20.0)
)

obs_l_dict = {
    k: obstacle_l_dict[k] for k in ['empty']
}
env = TransportationMixEnv(config, obs_l_dict)

model = SAC.load('model_ckp/pos_tol_2_angle_pi_corridor_10_20_reward_scale_10')
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
        print()
        obs, info = env.reset()
        print('Reward: ', acc_reward)
        acc_reward = 0
        print('Success rate: ', sum(success_l) / len(success_l))

