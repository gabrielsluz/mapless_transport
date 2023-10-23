# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.envs.navigation_env import NavigationEnvConfig, NavigationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from research_envs.b2PushWorld.NavigationWorld import NavigationWorldConfig, LaserHit
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

import cv2
import numpy as np
from Box2D import b2Vec2

# Potential Fields
def F_att(point, goal, k_att):
    att_force = k_att * (goal - point)
    if np.linalg.norm(att_force) > 30:
        att_force = 30 * att_force / np.linalg.norm(att_force)
    return att_force

def F_rep(point, obs_point, dist,  min_dist, k_rep, gamma):
    if dist > min_dist:
        return np.array([0.0, 0.0])
    
    theta = np.pi/4
    # Rotation 2D matrix for theta radians:
    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.cos(theta), np.sin(theta)]
    ])

    dist = max(dist, 1e-4)
    force = k_rep/dist**2 * (1/dist - 1/min_dist)**(gamma-1) * (point - obs_point)/dist
    return rot_mat @ force

def calc_direction():
    world = env.cur_env.world
    x_i = world.agent.agent_rigid_body.position.x
    y_i = world.agent.agent_rigid_body.position.y
    att_force = F_att(
        np.array([x_i, y_i]),
        np.array(world.goal),
        1.5
    )
    range_l, type_l, point_l = world.get_laser_readings_from_point(b2Vec2(x_i, y_i))
    rep_force = 0.0
    for i in range(len(range_l)):
        if type_l[i] != LaserHit.OBSTACLE: continue
        rep_force += F_rep(
            point=np.array([x_i, y_i]),
            obs_point=np.array(point_l[i]),
            dist=range_l[i],
            min_dist=5,
            k_rep=200,
            gamma=2
        )
    force = att_force + rep_force
    return force / np.linalg.norm(force)

def render():
    scene_buffer.PushFrame(env.render())
    scene_buffer.Draw()
    cv2.waitKey(1)
    # pass

config = NavigationEnvConfig(
    world_config= NavigationWorldConfig(
        obstacle_l = [],
        n_rays = 24,
        range_max = 25.0,
        agent_type = 'continuous'
    ),
    max_steps = 200,
    previous_obs_queue_len = 0
)
obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        'frame', 'horizontal_corridor', 'vertical_corridor',
        '4_circles_wide',
        '1_circle', '1_rectangle', '1_triangle',
        # 'circle_line', 'small_4_circles',
        # '4_circles', 'sparse_1', 'sparse_2',
        # '16_circles', '25_circles', '49_circles',
        # '1_circle', '1_rectangle', '1_triangle',
        # 'corridor', 'crooked_corridor',
        # 'small_U', 'small_G',
        # 'U', 'G'

        #'circle_line', 'small_4_circles', 'empty'
        #'small_4_circles', '16_circles', '25_circles', '49_circles',
        #'empty', 'circle_line', 'small_4_circles',
        #'1_circle', '1_rectangle', '1_triangle', 
        #'4_circles', '16_circles', 'corridor', 'crooked_corridor',
        #'sparse_1', 'sparse_2'
    ]
}
env = NavigationMixEnv(config, obs_l_dict)



scene_buffer = CvDrawBuffer(window_name="Simulation", resolution=(1024,1024))

render()
obs, info = env.reset()
acc_reward = 0
success_l = []
while True:
    action= calc_direction()
    obs, reward, terminated, truncated, info = env.step(action)
    acc_reward += reward
    render()
    if terminated or truncated:
        success_l.append(info['is_success'])
        obs, info = env.reset()
        print('Reward: ', acc_reward)
        acc_reward = 0
        print('Success rate: ', sum(success_l) / len(success_l))

