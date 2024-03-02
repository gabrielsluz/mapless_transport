# To execute from root folder
import sys
sys.path.append('.')

from research_envs.b2PushWorld.TransportationPoseWorld import TransportationWorldConfig
from research_envs.envs.transportation_pose_capsule_conditioned_env import TransportationEnvConfig, TransportationEnv, TransportationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from research_envs.envs.object_repo import object_desc_dict
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

import cv2
import json
import numpy as np
from Box2D import b2Vec2

# Key to action for continuous agent with action 
# spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
# theta = 2*pi * action[0] 
# direction = np.array([cos(theta), sin(theta)])
# force = action[1]
def key_to_action(key):
    action = None
    if key == 113: #q
        action = np.array([0.625, 1.0])
    elif key == 119: # w
        action = np.array([0.75, 1.0])
    elif key == 101: # e
        action = np.array([0.875, 1.0])
    elif key == 100: # d
        action = np.array([0.0, 1.0])
    elif key == 99: # c
        action = np.array([0.125, 1.0])
    elif key == 120: # x
        action = np.array([0.25, 1.0])
    elif key == 122: # z
        action = np.array([0.375, 1.0])
    elif key == 97: # a
        action = np.array([0.5, 1.0])
    return action


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
    cv2.waitKey(1)

if __name__ == "__main__":
    scene_buffer = CvDrawBuffer(window_name="Simulation", resolution=(1024,1024))
    config = TransportationEnvConfig(
        world_config= TransportationWorldConfig(
            obstacle_l = [],
            object_l=[object_desc_dict[0]],
            n_rays = 0,
            agent_type = 'continuous',
            max_force_length=1.0,
            min_force_length=0.0,
            goal_tolerance={'pos':1, 'angle':np.pi/18},
            max_obj_dist=8.0,
        ),
        max_steps=500,
        previous_obs_queue_len=0,
        reward_scale=1.0,
        corridor_width_range=(8.0, 8.0)
    )
    obs_l_dict = {
        k: obstacle_l_dict[k] 
        for k in [
            'empty'
        ]
    }
    env = TransportationMixEnv(config, obs_l_dict)

    print('Env created.')

    print('Obj radius: ', env.cur_env.world.obj.obj_radius)
    
    acc_reward = 0.0
    render()
    while True:
        # Input handling - requires a cv2 window running => env.render()
        dt = 1.0 / 60.0 #1.0 / 60.0
        key = 0xFF & cv2.waitKey(int(dt * 1000.0)) # Sets default key = 255
        if key == 27: break # Esc key
        action = key_to_action(key)
        if not action is None:
            observation, reward, terminated, truncated, info = env.step(action)
            acc_reward += reward
            render()
            # print('Pos:', env.cur_env.world.agent.agent_rigid_body.position)
            print(observation)
            print(observation.shape)
            print('Reward: ', reward)
            if terminated: 
                print('Terminated')
            if truncated:
                print('Truncated')
            print('Info: ', info)

            if env.cur_env.world.did_agent_collide():
                print('Agent collided with obstacle.')
            if env.cur_env.world.did_object_reach_goal():
                print('Agent reached goal.')

            if truncated or terminated:
                print("Accumulated reward: ", acc_reward)
                acc_reward = 0.0
                env.reset()