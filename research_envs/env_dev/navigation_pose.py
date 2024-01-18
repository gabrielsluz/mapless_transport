# To execute from root folder
import sys
sys.path.append('.')

from research_envs.b2PushWorld.NavigationPoseWorld import NavigationWorldConfig
from research_envs.envs.navigation_pose_env import NavigationEnvConfig, NavigationEnv, NavigationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

import cv2
import numpy as np

from Box2D import b2Vec2

def key_to_action(key):
    action = None
    if key == 113: #q
        action = np.array([0.625, 1.0, 0.25])
    elif key == 119: # w
        action = np.array([0.75, 1.0, 0.25])
    elif key == 101: # e
        action = np.array([0.875, 1.0, 0.0])
    elif key == 100: # d
        action = np.array([0.0, 1.0, 0.0])
    elif key == 99: # c
        action = np.array([0.125, 1.0, 0.0])
    elif key == 120: # x
        action = np.array([0.25, 1.0, 0.0])
    elif key == 122: # z
        action = np.array([0.375, 1.0, 0.0])
    elif key == 97: # a
        action = np.array([0.5, 1.0, 0.0])
    return action


def render():
    scene_buffer.PushFrame(env.render())
    scene_buffer.Draw()
    cv2.waitKey(1)

if __name__ == "__main__":
    scene_buffer = CvDrawBuffer(window_name="Simulation", resolution=(1024,1024))
    config = NavigationEnvConfig(
        world_config= NavigationWorldConfig(
            obstacle_l = [],
            n_rays = 24,
            range_max = 25.0,
            max_force_length=1.0,
            min_force_length=0.0,
            width=90.0,
            height=80.0,
            goal_tolerance={'pos': 2.0, 'angle': np.deg2rad(180)}
        ),
        max_steps=200,
        previous_obs_queue_len=0
    )
    obs_l_dict = {
        k: obstacle_l_dict[k] 
        for k in [
            'parallel_walls',
            # 'circle_line', 'small_4_circles',
            # '4_circles', 'sparse_1', 'sparse_2',
            # '1_circle', '1_rectangle', '1_triangle',
            # 'corridor', 'crooked_corridor',
            # '16_circles', '25_circles', '49_circles',
            # 'small_U', 'small_G',
            # 'U', 'G',
            # 'frame', 'horizontal_corridor', 'vertical_corridor',
            # '4_circles_wide'
        ]
    }
    env = NavigationMixEnv(config, obs_l_dict)
    # config = NavigationEnvConfig(
    #     world_config= NavigationWorldConfig(
    #         obstacle_l = [
    #             {'name':'Circle', 'pos':(5.0, 5.0), 'radius':2.0},
    #             {'name':'Circle', 'pos':(10.0, 10.0), 'radius':5.0},
    #             {'name':'Circle', 'pos':(35.0, 35.0), 'radius':2.0},
    #             {'name':'Circle', 'pos':(45.0, 35.0), 'radius':2.0},
    #             {'name':'Circle', 'pos':(5.0, 35.0), 'radius':4.0},
    #             {'name':'Rectangle', 'pos':(25.0, 25.0), 'height':10.0, 'width':2.0}
    #         ],
    #         n_rays = 8,
    #         range_max = 4.0
    #     ),
    #     max_steps=200
    # )
    #env = NavigationEnv(config)
    print('Env created.')
    
    render()
    while True:
        # Input handling - requires a cv2 window running => env.render()
        dt = 1.0 / 60.0 #1.0 / 60.0
        key = 0xFF & cv2.waitKey(int(dt * 1000.0)) # Sets default key = 255
        if key == 27: break # Esc key
        action = key_to_action(key)
        if not action is None:
            observation, reward, terminated, truncated, info = env.step(action)
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
            if env.cur_env.world.did_agent_reach_goal():
                print('Agent reached goal.')

            if truncated or terminated:
                env.reset()