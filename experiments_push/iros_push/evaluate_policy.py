"""
For a list of corridor_width, run N epsiodes and compute the success rate.
"""
# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.b2PushWorld.TransportationPoseWorld import TransportationWorldConfig
from research_envs.envs.transportation_pose_capsule_conditioned_env import TransportationEnvConfig, TransportationEnv
from research_envs.envs.object_repo import object_desc_dict
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

import cv2
import numpy as np
from stable_baselines3 import SAC

from tqdm import tqdm
import pandas as pd
import os


def render():
    screen = env.render()
    start_pos = env.start_obj_pos
    end_pos = env.world.goal['pos']

    start = env.world.worldToScreen((start_pos.x, start_pos.y))
    end = env.world.worldToScreen((end_pos.x, end_pos.y))
    screen = cv2.line(screen, start, end, color=(0, 0, 255), thickness=2)

    # Draw a line ax + b = y
    corr_line = env.corr_line
    a,b = corr_line[0], corr_line[1]

    # Find the paralel lines
    w = env.corridor_width
    angle_line_to_x = np.arctan2(a, 1)

    start_pos = env.start_obj_pos
    start_upper = (
        start_pos.x + w * np.cos(np.pi/2 + angle_line_to_x),
        start_pos.y + w * np.sin(np.pi/2 + angle_line_to_x)
    )
    start_lower = (
        start_pos.x + w * np.cos(-np.pi/2 + angle_line_to_x),
        start_pos.y + w * np.sin(-np.pi/2 + angle_line_to_x)
    )

    end_pos = env.world.goal['pos']
    end_upper = (
        end_pos.x + w * np.cos(np.pi/2 + angle_line_to_x),
        end_pos.y + w * np.sin(np.pi/2 + angle_line_to_x)
    )
    end_lower = (
        end_pos.x + w * np.cos(-np.pi/2 + angle_line_to_x),
        end_pos.y + w * np.sin(-np.pi/2 + angle_line_to_x)
    )

    start = env.world.worldToScreen(start_upper)
    end = env.world.worldToScreen(end_upper)
    screen = cv2.line(screen, start, end, color=(0, 0, 255), thickness=2)

    start = env.world.worldToScreen(start_lower)
    end = env.world.worldToScreen(end_lower)
    screen = cv2.line(screen, start, end, color=(0, 0, 255), thickness=2)

    # Circles
    screen_pos = env.world.worldToScreen(start_pos)
    cv2.circle(screen, screen_pos, int(w*env.world.pixels_per_meter), color=(0, 0, 255), thickness=2)

    screen_pos = env.world.worldToScreen(end_pos)
    cv2.circle(screen, screen_pos, int(w*env.world.pixels_per_meter), color=(0, 0, 255), thickness=2)


    scene_buffer.PushFrame(screen)
    scene_buffer.Draw()
    cv2.waitKey(100)

###################### OBJECT ID ######################
obj_id = int(sys.argv[1])
exp_name = 'obj_' + str(obj_id)

config = TransportationEnvConfig(
    world_config= TransportationWorldConfig(
        obstacle_l = [],
        object_l=[object_desc_dict[obj_id]],
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
    corridor_width_range = (12.0, 12.0)
)
env = TransportationEnv(config)
model = SAC.load('model_ckp/'+exp_name+'/best_model')
print(model.policy)

capsule_width_l = np.linspace(10, 20, 11)
ep_per_width = 500

scene_buffer = CvDrawBuffer(window_name="Simulation", resolution=(1024,1024))
render_bool = False

res_d = {'capsule_width': [], 'success': [], 'time_steps': [], 'acc_reward': []}

for w in tqdm(capsule_width_l):
    env.corridor_width_range = (w, w)
    env.reset()
    for _ in range(ep_per_width):
        obs, info = env.reset()
        acc_reward = 0
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            acc_reward += reward
            if render_bool: render()
            if terminated or truncated:
                res_d['capsule_width'].append(w)
                res_d['acc_reward'].append(acc_reward)
                res_d['success'].append(info['is_success'])
                res_d['time_steps'].append(env.step_count)
                break

df = pd.DataFrame(res_d)

# Create dir 
if not os.path.exists('eval_results'): 
    os.makedirs('eval_results') 

df.to_csv('eval_results/' + exp_name + '.csv', index=False)