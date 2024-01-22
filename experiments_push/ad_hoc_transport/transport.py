# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.envs.transportation_pose_corridor_env import TransportationEnvConfig, TransportationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from research_envs.b2PushWorld.TransportationPoseWorld import TransportationWorldConfig
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

import cv2

from shapely import LineString, Polygon, Point
from Box2D import b2Vec2, b2Transform

from stable_baselines3 import SAC

import random
import math
import numpy as np
import types

forbidden_polys = []
sg_candidates = []
is_valid_sg = []
last_theta = 0.0
chosen_sg = None

def adjusted_check_death(self):
    return self.world.did_agent_collide() or self.world.did_object_collide()

def adjusted_check_success(env):
    self = env.world
    reached_pos = (final_goal['pos'] - self.obj.obj_rigid_body.position).length < self.goal_tolerance['pos']

    # Calculate the angle between the object and the goal
    angle = self.obj.obj_rigid_body.angle % (2*np.pi)
    if angle < 0.0: angle += 2*np.pi
    angle_diff = final_goal['angle'] - angle
    if angle_diff > np.pi:
        angle_diff -= 2*np.pi
    elif angle_diff < -np.pi:
        angle_diff += 2*np.pi
    reached_angle = abs(angle_diff) < self.goal_tolerance['angle']

    return reached_pos and reached_angle

def check_subgoal_sucess(env):
    self = env.world
    reached_pos = (self.goal['pos'] - self.obj.obj_rigid_body.position).length < subgoal_tolerance['pos']

    # Calculate the angle between the object and the goal
    angle = self.obj.obj_rigid_body.angle % (2*np.pi)
    if angle < 0.0: angle += 2*np.pi
    angle_diff = self.goal['angle'] - angle
    if angle_diff > np.pi:
        angle_diff -= 2*np.pi
    elif angle_diff < -np.pi:
        angle_diff += 2*np.pi
    reached_angle = abs(angle_diff) < subgoal_tolerance['angle']

    return reached_pos and reached_angle

def set_new_goal(self, new_goal={'pos':b2Vec2(0,0), 'angle': 0.0}):    
    self.world.goal = new_goal
    self.step_count = 0
    self.prev_action_queue.clear()
    # self.prev_obs_queue.clear()
    self.last_dist = self.world.object_to_goal_vector().length
    self.last_orient_error = self.world.distToOrientation()/ np.pi
    self.start_obj_pos = b2Vec2(self.world.obj.obj_rigid_body.position)
    self.corr_line = [
        (self.world.goal['pos'].y - self.start_obj_pos.y) / (self.world.goal['pos'].x - self.start_obj_pos.x),
        self.start_obj_pos.y - self.start_obj_pos.x * (self.world.goal['pos'].y - self.start_obj_pos.y) / (self.world.goal['pos'].x - self.start_obj_pos.x)
    ]
    return self._gen_observation(), {}

def _gen_subgoal_candidate(self, angle_range=(0, 2*np.pi)):
    # Gen subgoals in a range radius from the objects position
    rand_dist = random.uniform(min_subgoal_pos_dist, max_subgoal_pos_dist)
    rand_rad = random.uniform(angle_range[0], angle_range[1])
    subgoal_pos = [
        self.world.obj.obj_rigid_body.position.x + rand_dist * np.cos(rand_rad),
        self.world.obj.obj_rigid_body.position.y + rand_dist * np.sin(rand_rad)
    ]
    # Angle
    cur_angle = math.fmod(self.world.obj.obj_rigid_body.angle, 2*math.pi)
    if cur_angle < 0.0: cur_angle += 2*math.pi
    rand_angle = random.uniform(
        cur_angle - max_subgoal_angle_dist, 
        cur_angle + max_subgoal_angle_dist)

    return {'pos': subgoal_pos, 'angle': rand_angle}

def _gen_rectangle_from_center_line(self, start_p, end_p, w):
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

def _calc_goal_dist(subgoal, final_goal):
    pos_dist = (final_goal['pos'] - b2Vec2(subgoal['pos'])).length
    pos_dist = pos_dist / max_subgoal_pos_dist

    # Calculate the angle between the object and the goal
    angle = subgoal['angle'] % (2*np.pi)
    if angle < 0.0: angle += 2*np.pi
    angle_diff = final_goal['angle'] - angle
    if angle_diff > np.pi:
        angle_diff -= 2*np.pi
    elif angle_diff < -np.pi:
        angle_diff += 2*np.pi
    angle_dist = abs(angle_diff) / np.pi

    # Normalize pos_dist and add to angle_dist
    return pos_dist + angle_dist

# Ad Hoc Navigation
"""
Subgoal selection high level:

default subgoal => final_goal

For range in (desired_range, complement):
    Sample candidates on desired range.
    For each corridor_width in descending order:
        filtered_candidates = filter for corridor width
        if len(filtered_candidates) > 0:
            return the best according to ranking metric.
return final_goal
""" 
def find_best_subgoal(env):
    global forbidden_polys, sg_candidates, is_valid_sg, chosen_sg, last_theta
    self = env
    # Find the forbidden polygons => likely to have obstacles
    forbidden_polys = []
    range_l, _, point_l = self.world.get_laser_readings()
    agent_pos = self.world.obj.obj_rigid_body.position
    # Always forward
    for i, p in enumerate(point_l):
        next_i = (i+1) % len(point_l)
        next_p = point_l[next_i]
        next_p_points = [(next_p[0], next_p[1])]
        if range_l[next_i] < self.world.range_max:
            dir_vec = (next_p - agent_pos)
            dir_vec.Normalize()
            next_p_end = agent_pos + dir_vec * self.world.range_max
            next_p_points = [
                (next_p_end[0], next_p_end[1]),
                (next_p[0], next_p[1])
            ]
        # Obstacle detected in i
        if range_l[i] < self.world.range_max:
            # Assemble quadrilateral with last point and the next one
            # Start point = p
            # End point: line from agent to p extended to range_max
            dir_vec = (p - agent_pos)
            dir_vec.Normalize()
            end_p = agent_pos + dir_vec * self.world.range_max
            forbidden_polys.append(
                Polygon([
                    *next_p_points,
                    (p[0], p[1]), 
                    (end_p[0], end_p[1])
                ])
            )
        # Obstacle detected in the next point
        elif range_l[next_i] < self.world.range_max:
            forbidden_polys.append(
                Polygon([
                    *next_p_points,
                    (p[0], p[1])
                ])
            )

    # Find the range of angles to sample
    desired_range = (last_theta - max_theta_change, last_theta + max_theta_change)
    complement_range = (desired_range[1], desired_range[0] + 2*np.pi)
    range_l = [desired_range, complement_range]

    # Default subgoal
    min_sg = {'pos' : final_goal['pos'], 'angle': final_goal['angle']}
    for r in range_l:
        found_valid_sg = False
        sg_candidates = [_gen_subgoal_candidate(self, r) for _ in range(n_sampled_candidates)]
        if (final_goal['pos'] - agent_pos).length <= max_subgoal_pos_dist:
            sg_candidates.append(
                {'pos' : final_goal['pos'], 'angle': final_goal['angle']})
        for c_w in corridor_width_l:
            min_dist = None
            transform_matrix = b2Transform()
            transform_matrix.SetIdentity()
            is_valid_sg = []
            for sg in sg_candidates:
                # Create polygon on pos and angle, based on the object
                # transform_matrix.Set(sg['pos'], sg['angle'])
                # vertices = [(transform_matrix * v) for v in agent_vertices]
                # robot_body = Polygon(vertices)
                robot_body = Point(sg['pos'][0], sg['pos'][1]).buffer(10.0)
                # Union with the robot body
                corridor = Polygon(
                    _gen_rectangle_from_center_line(self, (self.world.obj.obj_rigid_body.position.x, self.world.obj.obj_rigid_body.position.y), sg['pos'], c_w))
                poly = corridor.union(robot_body)
                # Check if the subgoal is in the forbidden lines
                is_valid = True
                for f_p in forbidden_polys:
                    if poly.intersects(f_p):
                        is_valid = False
                        break
                is_valid_sg.append(is_valid)
                if is_valid:
                    dist = _calc_goal_dist(sg, final_goal)
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                        min_sg = sg
            # If found a valid subgoal, break
            if sum(is_valid_sg) > 0:
                found_valid_sg = True
                break
        if found_valid_sg:
            break

    dir_x = min_sg['pos'][0] - self.world.obj.obj_rigid_body.position.x
    dir_y = min_sg['pos'][1] - self.world.obj.obj_rigid_body.position.y
    last_theta = math.atan2(dir_y, dir_x)
    # Take a small step towards the subgoal
    dir_vec = b2Vec2(dir_x, dir_y)
    if dir_vec.length > subgoal_step_len:
        dir_vec.Normalize()
        dir_vec = dir_vec * subgoal_step_len

    min_sg['pos'] = (dir_vec.x + self.world.obj.obj_rigid_body.position.x, dir_vec.y + self.world.obj.obj_rigid_body.position.y)

    chosen_sg = min_sg

    return min_sg

    # dir_x = min_sg['pos'][0] - agent_pos.x
    # dir_y = min_sg['pos'][1] - agent_pos.y
    # theta = math.atan2(dir_y, dir_x) / (2*math.pi)
    # force = 1.0
    # last_theta = theta * 2*math.pi

    # # Calc angle proportional to the distance
    # ratio = force / b2Vec2(dir_x, dir_y).length
    # cur_angle = math.fmod(self.world.obj.obj_rigid_body.angle, 2*math.pi)
    # if cur_angle < 0.0: cur_angle += 2*math.pi
    # final_angle = cur_angle + ratio*(min_sg['angle'] - cur_angle)
    # final_angle = final_angle / (2*math.pi)

    # return np.array([theta, force, final_angle])

def adjust_obs(obs):
    return obs[72:]


def render():
    screen = env.render()
    self = env.cur_env

    #Draw the forbidden_polys
    # for poly in forbidden_polys:
    #     vertices = list(poly.exterior.coords)
    #     vertices = [self.world.worldToScreen(v) for v in vertices]
    #     cv2.fillPoly(screen, [np.array(vertices)], (255, 0, 0))

    # print('Num is_valid:', sum(is_valid_sg))
    # If num_is_valid = 0, stop the screen for a little
    if sum(is_valid_sg) == 0:
        print('No valid subgoals')
        cv2.waitKey(1000)
    for i in range(len(sg_candidates)):
        sg = sg_candidates[i]
        sg = self.world.worldToScreen(sg['pos'])
        if is_valid_sg[i]:
            cv2.circle(screen, sg, 5, (0, 255, 0), -1)
        else:
            cv2.circle(screen, sg, 5, (255, 0, 0), -1)
    
    # Draw the chosen_sg corridor
    if chosen_sg is not None:
        # corridor = _gen_rectangle_from_center_line(self, (self.world.obj.obj_rigid_body.position.x, self.world.obj.obj_rigid_body.position.y), chosen_sg['pos'], corridor_width)
        corridor = _gen_rectangle_from_center_line(self, (self.start_obj_pos.x, self.start_obj_pos.y), chosen_sg['pos'], corridor_width)
        corridor = [self.world.worldToScreen(v) for v in corridor]
        # cv2.fillPoly(screen, [np.array(corridor)], (0, 255, 0))
        cv2.polylines(screen, [np.array(corridor)], isClosed=True, color=(0, 255, 0), thickness=4)

    # Draw final goal in yellow
    self.world.obj.DrawInPose(
        final_goal['pos'], final_goal['angle'], self.world.pixels_per_meter, screen, (0, 255, 255), -1)
    self.world.drawArrow(screen, final_goal['pos'], final_goal['angle'], 10, (0, 255, 255))

    frame_l.append(screen)
    scene_buffer.PushFrame(screen)
    scene_buffer.Draw()
    cv2.waitKey(1)

def save_video(frame_l):
    global video_counter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    f_name = 'videos/video_' + str(video_counter) + '.mp4'
    out = cv2.VideoWriter(f_name, fourcc, 20, (frame_l[0].shape[1], frame_l[0].shape[0]))
    for frame in frame_l:
        frame = frame.astype(np.uint8)
        out.write(frame)
    out.release()

    video_counter += 1

object_desc = {'name': 'Rectangle', 'height': 10.0, 'width': 5.0}
print(object_desc)

config = TransportationEnvConfig(
    world_config= TransportationWorldConfig(
        obstacle_l = [],
        object_l=[
            object_desc
            ],
        n_rays = 72,
        range_max = 25.0,
        agent_type = 'continuous',
        max_force_length=5.0,
        min_force_length=0.1,
        width=100.0,
        height=100.0,
        goal_tolerance={'pos':2, 'angle':np.pi/18},
        max_obj_dist=10.0
    ),
    max_steps = 100,
    previous_obs_queue_len = 0,
    reward_scale=10.0,
    max_corr_width=10.0
)

obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        'parallel_walls'
        # 'empty', 'frame', 'horizontal_corridor', 'vertical_corridor', '4_circles_wide',
        # '1_circle', '1_rectangle', '1_triangle',
        # 'circle_line', 'small_4_circles',
        # '4_circles', 'sparse_1', 'sparse_2',
    ]
}
env = TransportationMixEnv(config, obs_l_dict)
env.cur_env._check_death = types.MethodType(adjusted_check_death, env.cur_env)
env.cur_env._check_success = types.MethodType(adjusted_check_success, env.cur_env)

# Set parameters
min_subgoal_pos_dist = 15.0
max_subgoal_pos_dist = 15.0
subgoal_step_len = 10.0
corridor_width = 10.0
corridor_width_l = [10, 8, 6]
max_subgoal_angle_dist = np.pi/8
max_theta_change = np.pi/4
update_subgoal_every = 10
n_sampled_candidates = 100

subgoal_tolerance = {'pos':3, 'angle':np.pi/4}

agent_vertices = env.cur_env.world.obj.obj_rigid_body.fixtures[0].shape.vertices

# Goal
final_goal = {'pos':b2Vec2(80, 20), 'angle': 3*np.pi/2}

# Load agent
model = SAC.load('model_ckp/progress_sac_rectangle_tolerance_pi18_pos_tol_2_reward_scale_10_corridor_full_death_width_10')

# Rendering
scene_buffer = CvDrawBuffer(window_name="Simulation", resolution=(1024,1024))
frame_l = []
video_counter = 0

counter = 0
render()
obs, info = env.reset()
acc_reward = 0
success_l = []
while True:
    # Find the direction, take a simple step towards.
    # Check if subgoal has been reached
    if check_subgoal_sucess(env.cur_env): counter = 0
    if sum(is_valid_sg) == 0: counter  = 0
    if counter % update_subgoal_every == 0:
        min_sg = find_best_subgoal(env.cur_env)
        set_new_goal(env.cur_env, new_goal={'pos':b2Vec2(min_sg['pos']), 'angle': min_sg['angle']})
        counter = 0
    counter += 1
    
    obs = adjust_obs(obs)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    acc_reward += reward
    render()
    if terminated or truncated:
        success_l.append(info['is_success'])
        # Save video
        # if not info['is_success']:
        #     save_video(frame_l)
        save_video(frame_l)
        frame_l = []

        final_goal['angle'] = random.uniform(0, 2*np.pi)

        obs, info = env.reset()

        counter = 0

        env.cur_env._check_death = types.MethodType(adjusted_check_death, env.cur_env)
        env.cur_env._check_success = types.MethodType(adjusted_check_success, env.cur_env)

        print('Reward: ', acc_reward)
        acc_reward = 0
        print('Success rate: ', sum(success_l) / len(success_l))

