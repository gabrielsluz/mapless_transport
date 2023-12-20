# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.b2PushWorld.TransportationPoseWorld import TransportationWorldConfig
from research_envs.envs.transportation_pose_env import TransportationEnvConfig, TransportationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

from Box2D import b2Vec2

import cv2
import random
import json
import numpy as np
import math
import pandas as pd
from stable_baselines3 import PPO, SAC

config = TransportationEnvConfig(
    world_config= TransportationWorldConfig(
        obstacle_l = [],
        object_l=[
            # {'name': 'MultiPolygons', 'poly_vertices_l':[[[0, 0], [0, 4], [12, 4], [12, 0]], [[0, 4], [0, 8], [4, 8], [4,4]]]}
            # {'name': 'Rectangle', 'height': 10.0, 'width': 5.0}
            {'name': 'Polygon', 'vertices':[[0, 0], [6, 9], [16, 0]]}
            # {'name': 'MultiPolygons', 'poly_vertices_l':[
            #         [[0, 0], [0, 4], [2, 4], [4, 2], [4, 0]],
            #         [[0, 0], [0, 2], [-6, 2], [-6, 0]],
            #         [[0, 0], [-4, -6], [0, -4]],

            #         [[0, 0], [0, -4], [4, -4]],
            #         [[0, 0], [2, -2], [4, 0]]
            #     ]}
            # {
            #     'name': 'MultiPolygons',
            #     'poly_vertices_l': json.load(
            #         open('../../research_envs/obj_utils/polygons/tentacle_multi.json', 'r')
            #     )['polygons']
            # }
        ],
        n_rays = 0,
        agent_type = 'continuous',
        max_force_length=5.0,
        min_force_length=0.25,
        max_obj_dist=15.0,
        goal_tolerance={'pos':2, 'angle':np.pi/36}
    ),
    max_steps = 2000,
    previous_obs_queue_len = 0
)

obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        'empty'
    ]
}
env = TransportationMixEnv(config, obs_l_dict)

exp_name = 'progress_sac_triangle_tolerance_pi36_pos_tol_2_reward_scale_20'
# model = PPO.load("model_ckp/L_min_force_length_025")
model = SAC.load("model_ckp/" + exp_name)
print(model.policy)

n_episodes = 500

initial_distance_l = [25]#[5.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0]
init_d_id = 0

# CODE FOR EVALUATION

# Corridor width
def distance_point_to_line(p, p1, p2):
    # return the distance from p to the line defined by p1 and p2
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    return abs((p2.x - p1.x) * (p1.y - p.y) - (p1.x - p.x) * (p2.y - p1.y)) / ((p2-p1).length)

def max_distance_object_to_line(obj, p1, p2):
    # return the maximum distance from obj to the line defined by p1 and p2
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    # Compute the distance from each vertex of the object to the line
    # Return the maximum
    max_dist = 0
    body = obj.obj_rigid_body
    for f_i in range(len(body.fixtures)):
        vertices = [(body.transform * v) for v in body.fixtures[f_i].shape.vertices]
        for v in vertices:
            max_dist = max(max_dist, distance_point_to_line(v, p1, p2))
    return max_dist

# def max_distance_object_to_line(obj, p1, p2):
#     # return the maximum distance from circle to the line defined by p1 and p2
#     # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
#     # Return the maximum
#     v = obj.obj_rigid_body.position
#     return distance_point_to_line(v, p1, p2) + radius

# Reset goal at specific distance from object
def reset_goal_at_distance_from_object(env, distance):
    env.reset() # MixEnv.reset()
    env = env.cur_env

    env.world.reset()
    rand_rad = random.uniform(0, 2*math.pi)
    env.world.goal['pos'].x = env.world.obj.obj_rigid_body.position.x + distance * math.cos(rand_rad)
    env.world.goal['pos'].y = env.world.obj.obj_rigid_body.position.y + distance * math.sin(rand_rad)

    env.step_count = 0

    env.prev_action_queue.clear()
    env.last_dist = env.world.object_to_goal_vector().length
    return env._gen_observation(), {}

scene_buffer = CvDrawBuffer(window_name="Simulation", resolution=(1024,1024))
def render():
    # pass
    scene_buffer.PushFrame(env.render())
    scene_buffer.Draw()
    cv2.waitKey(1)

cur_ep = 0
ep_cnt = 0
result_d = {'id': [], 'success': [], 'corridor_width': [], 'trajectory_efficiency': [], 'init_distance': []}

render()
obs, info = reset_goal_at_distance_from_object(env, initial_distance_l[init_d_id])

cur_length = 0
start_pos = b2Vec2(env.cur_env.world.obj.obj_rigid_body.position)
last_pos = b2Vec2(env.cur_env.world.obj.obj_rigid_body.position)
max_corridor_width = 0
acc_reward = 0
while True:
    action, _states = model.predict(obs, deterministic=False)
    # action = env.cur_env.world.agent.GetRandomValidAction()
    obs, reward, terminated, truncated, info = env.step(action)
    render()

    acc_reward += reward
    cur_pos = env.cur_env.world.obj.obj_rigid_body.position
    # print(cur_pos, last_pos)
    # print((cur_pos - last_pos).length)
    cur_length += (cur_pos - last_pos).length
    last_pos = b2Vec2(cur_pos)
    max_corridor_width = max(
        max_corridor_width, 
        max_distance_object_to_line(env.cur_env.world.obj, start_pos, b2Vec2(env.cur_env.world.goal['pos'])))

    if terminated or truncated:
        result_d['id'].append(ep_cnt)
        result_d['success'].append(info['is_success'])
        result_d['corridor_width'].append(max_corridor_width)
        start_dist = (last_pos - start_pos).length
        if start_dist > 0:
            result_d['trajectory_efficiency'].append(cur_length / (start_dist))
        else:
            result_d['trajectory_efficiency'].append(None)
        result_d['init_distance'].append(initial_distance_l[init_d_id])
        ep_cnt += 1

        cur_ep += 1
        if cur_ep >= n_episodes:
            print('Done with init distance: ', initial_distance_l[init_d_id])
            init_d_id += 1
            cur_ep = 0
            if init_d_id >= len(initial_distance_l):
                break

        obs, info = reset_goal_at_distance_from_object(env, initial_distance_l[init_d_id])
        # print('Reward: ', acc_reward)
        # print('Max Corridor Width: ', max_corridor_width)
        # print('Trajectory Length: ', cur_length)
        # print('Trajectory Efficiency: ', cur_length / (start_dist))
        # print('Success rate: ', sum(result_d['success']) / len(result_d['success']))
        # print()
        acc_reward = 0
        cur_length = 0
        start_pos = b2Vec2(env.cur_env.world.obj.obj_rigid_body.position)
        last_pos = b2Vec2(env.cur_env.world.obj.obj_rigid_body.position)
        max_corridor_width = 0
        

df = pd.DataFrame(result_d)
df.to_csv('results/' +exp_name+ '.csv', index=False)