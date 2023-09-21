# To execute from root folder
import sys
sys.path.append('.')

from research_envs.b2PushWorld.NavigationWorld import NavigationWorldConfig
from research_envs.envs.navigation_env import NavigationEnvConfig, NavigationEnv, NavigationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

import cv2

def key_to_action(key):
    action = -1
    if key == 97: # a
        action = 4
    elif key == 115: # s
        action = 2
    elif key == 100: # d
        action = 0
    elif key  == 119: # w
        action = 6
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
            n_rays = 36,
            range_max = 8.0
        ),
        max_steps=200
    )
    obs_l_dict = {
        k: obstacle_l_dict[k] 
        for k in obstacle_l_dict.keys()
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
        if action != -1:
            observation, reward, terminated, truncated, info = env.step(action)
            render()
            print('Pos:', env.cur_env.world.agent.agent_rigid_body.position)
            # print(observation)
            print('Reward: ', reward)
            if terminated: 
                print('Terminated')
            if truncated:
                print('Truncated')

            if env.cur_env.world.did_agent_collide():
                print('Agent collided with obstacle.')
                env.reset()
            if env.cur_env.world.did_agent_reach_goal():
                print('Agent reached goal.')
                env.reset()