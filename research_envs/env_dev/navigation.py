# To execute from root folder
import sys
sys.path.append('.')

from research_envs.b2PushWorld.NavigationWorld import NavigationWorld, NavigationWorldConfig
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
    scene_buffer.PushFrame(world.drawToBufferWithLaser())
    scene_buffer.Draw()
    cv2.waitKey(1)

if __name__ == "__main__":
    scene_buffer = CvDrawBuffer(window_name="Simulation", resolution=(1024,1024))
    config = NavigationWorldConfig(
        obstacle_l = [
            {'name':'Circle', 'pos':(5.0, 5.0), 'radius':2.0},
            {'name':'Circle', 'pos':(35.0, 35.0), 'radius':2.0},
            {'name':'Circle', 'pos':(5.0, 35.0), 'radius':4.0},
            {'name':'Rectangle', 'pos':(25.0, 25.0), 'height':10.0, 'width':2.0}
        ]
    )
    world = NavigationWorld(config)
    print('World created.')
    
    render()
    while True:
        # Input handling - requires a cv2 window running => env.render()
        dt = 1.0 / 60.0 #1.0 / 60.0
        key = 0xFF & cv2.waitKey(int(dt * 1000.0)) # Sets default key = 255
        if key == 27: break # Esc key

        action = key_to_action(key)
        if action != -1:
            # print('World update.')
            # print(world.get_laser_readings())
            world.take_action(action)
            render()

            if world.did_agent_collide():
                print('Agent collided with obstacle.')
                world.reset()
            if world.did_agent_reach_goal():
                print('Agent reached goal.')
                world.reset()