# To execute from root folder
import sys
sys.path.append('.')

from research_envs.b2PushWorld.TransportationPoseWorld import TransportationWorldConfig
from research_envs.envs.transportation_pose_corridor_env import TransportationEnvConfig, TransportationEnv, TransportationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
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
    # Draw a line ax + b = y
    corr_line = env.cur_env.corr_line
    a,b = corr_line[0], corr_line[1]

    # Find the paralel lines
    w = env.cur_env.max_corr_width
    angle_line_to_x = np.arctan2(a, 1)
    # Upper line:
    x = w * np.cos(np.pi/2 + angle_line_to_x)
    y = w * np.sin(np.pi/2 + angle_line_to_x) + b
    high_b = y - a * x

    # Lower line:
    x = w * np.cos(-np.pi/2 + angle_line_to_x)
    y = w * np.sin(-np.pi/2 + angle_line_to_x) + b
    low_b = y - a * x

    print(a,b, low_b, high_b)

    start = env.cur_env.world.worldToScreen(
        (0, low_b)
    )
    end = env.cur_env.world.worldToScreen(
        (100, 100*a + low_b)
    )
    screen = cv2.line(screen, start, end, color=(0, 0, 255), thickness=2)

    start = env.cur_env.world.worldToScreen(
        (0, high_b)
    )
    end = env.cur_env.world.worldToScreen(
        (100, 100*a + high_b)
    )
    screen = cv2.line(screen, start, end, color=(0, 0, 255), thickness=2)

    scene_buffer.PushFrame(screen)
    scene_buffer.Draw()
    cv2.waitKey(1)

if __name__ == "__main__":
    scene_buffer = CvDrawBuffer(window_name="Simulation", resolution=(1024,1024))

    config = TransportationEnvConfig(
        world_config= TransportationWorldConfig(
            obstacle_l = [],
            object_l=[
                # {'name':'Circle', 'radius':4.0},
                # Triangle using PolygonalObj:
                # {'name': 'Polygon', 'vertices':[[-4, -2], [4, -2], [0, 6]]},
                # {'name': 'Polygon', 'vertices':[[0, 0], [6, 9], [16, 0]]},
                {'name': 'Rectangle', 'height': 10.0, 'width': 5.0}
                # {'name': 'ConcavePolygon', 'vertices':[[-4, -2], [4, -2], [4, 6], [2, 6], [2, 2], [-4, 2]]}
                # {'name': 'MultiPolygons', 'poly_vertices_l':[[[0, 0], [0, 4], [12, 4], [12, 0]], [[0, 4], [0, 8], [4, 8], [4,4]]]}
                # {'name': 'MultiPolygons', 'poly_vertices_l':[
                #     [[0, 0], [0, 3], [6, 3], [3, 0]], 
                #     [[0, 3], [0, 6], [3, 3]],

                #     [[0, 0], [0, 3], [-6, 3], [-3, 0]], 
                #     [[0, 3], [0, 6], [-3, 3]],

                #     [[0, 0], [0, -3], [6, -3], [3, 0]], 
                #     [[0, -3], [0, -6], [3, -3]],

                #     [[0, 0], [0, -3], [-6, -3], [-3, 0]], 
                #     [[0, -3], [0, -6], [-3, -3]],
                #     ]}
                # Bizzarrer shape
                # {'name': 'MultiPolygons', 'poly_vertices_l':[
                #     [[0, 0], [0, 4], [2, 4], [4, 2], [4, 0]],
                #     [[0, 0], [0, 2], [-6, 2], [-6, 0]],
                #     [[0, 0], [-4, -6], [0, -4]],

                #     [[0, 0], [0, -4], [4, -4]],
                #     [[0, 0], [2, -2], [4, 0]]
                # ]}
                # Tentacle
                # {
                #     'name': 'MultiPolygons',
                #     'poly_vertices_l': json.load(
                #         open('research_envs/obj_utils/polygons/tentacle_multi.json', 'r')
                #     )['polygons']
                # }

            ],
            n_rays = 0,
            agent_type = 'continuous',
            max_force_length=1.0,
            min_force_length=0.0,
            goal_tolerance={'pos':2, 'angle':np.pi/18},
            max_obj_dist=10.0,
        ),
        max_steps=500,
        previous_obs_queue_len=0,
        reward_scale=1.0,
        max_corr_width=10.0
    )
    obs_l_dict = {
        k: obstacle_l_dict[k] 
        for k in [
            'empty'
        #     'empty', 'frame', 'horizontal_corridor', 'vertical_corridor','4_circles_wide',
        # '1_circle', '1_rectangle', '1_triangle'
            # 'circle_line', 'small_4_circles',
            #'4_circles', 'sparse_1', 'sparse_2',
            # '1_circle', '1_rectangle', '1_triangle',
            #'corridor', 'crooked_corridor',
            # '16_circles', '25_circles', '49_circles',
            # '1_circle', '1_rectangle', '1_triangle',
            # 'corridor', 'crooked_corridor',
            # '16_circles', '25_circles', '49_circles',
            # 'small_U', 'small_G',
            # 'U', 'G'
        ]
    }
    env = TransportationMixEnv(config, obs_l_dict)

    # config = TransportationEnvConfig(
    #     world_config= TransportationWorldConfig(
    #         obstacle_l = obstacle_l_dict['1_circle'],
    #         n_rays = 24,
    #         range_max = 25.0
    #     ),
    #     max_steps=200
    # )
    # env = TransportationEnv(config)
    print('Env created.')
    
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