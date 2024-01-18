# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.envs.navigation_env import NavigationEnvConfig, NavigationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from research_envs.b2PushWorld.NavigationWorld import NavigationWorldConfig
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

import cv2

from shapely import LineString, Polygon, Point
from Box2D import b2Vec2, b2Transform

import random
import math
import numpy as np

forbidden_polys = []
sg_candidates = []
is_valid_sg = []
last_theta = 0.0
chosen_sg = None

def _gen_subgoal_candidate(self):
    # Gen subgoals in a range radius from the objects position
    rand_dist = random.uniform(min_subgoal_pos_dist, max_subgoal_pos_dist)
    # neg = 1 if random.randint(0, 1) == 0 else -1
    # rand_dist = neg * rand_dist
    # rand_rad = random.uniform(0, 2*np.pi)
    rand_rad = random.uniform(last_theta - max_theta_change, last_theta + max_theta_change)

    subgoal_pos = [
        self.world.agent.agent_rigid_body.position.x + rand_dist * np.cos(rand_rad),
        self.world.agent.agent_rigid_body.position.y + rand_dist * np.sin(rand_rad)
    ]

    return subgoal_pos

def _gen_rectangle_from_center_line(self, start_p, end_p):
    # Creates a rectangle from the center line (start_p, end_p)
    # with width = corridor_width
    # Returns a list of 4 points in counter-clockwise order
    w = corridor_width
    line_angle = np.arctan2(end_p[1] - start_p[1], end_p[0] - start_p[0])
    vec = w * np.array([np.cos(line_angle + np.pi/2), np.sin(line_angle + np.pi/2)])
    return [
        (start_p[0] + vec[0], start_p[1] + vec[1]),
        (start_p[0] - vec[0], start_p[1] - vec[1]),
        (end_p[0] - vec[0], end_p[1] - vec[1]),
        (end_p[0] + vec[0], end_p[1] + vec[1])
    ]

# Ad Hoc Navigation
def find_best_action(env):
    global forbidden_polys, sg_candidates, is_valid_sg, chosen_sg, last_theta
    self = env
    # Find the forbidden lines => likely to have obstacles
    forbidden_polys = []
    range_l, _, point_l = self.world.get_laser_readings()
    agent_pos = self.world.agent.agent_rigid_body.position
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

    sg_candidates = [_gen_subgoal_candidate(self) for _ in range(100)]
    # If goal is inside the range, return it as min_sg
    if (self.world.goal - self.world.agent.agent_rigid_body.position).length <= max_subgoal_pos_dist:
        sg_candidates.append(self.world.goal)

    # Check if valid:
    min_dist = None
    min_sg = sg_candidates[0]
    # transform_matrix = b2Transform()
    # transform_matrix.SetIdentity()
    is_valid_sg = []
    for sg in sg_candidates:
        # Create polygon on pos and angle, based on the object
        # transform_matrix.Set(sg[0], sg[1])
        # vertices = [(transform_matrix * v) for v in self.obj_vertices]
        # poly = Polygon(vertices)
        # Union with the robot body
        robot_body = Point(sg).buffer(1.5*self.world.agent.agent_radius)
        corridor = Polygon(
            _gen_rectangle_from_center_line(self, (agent_pos.x, agent_pos.y), sg))
        poly = corridor.union(robot_body)
        # Check if the subgoal is in the forbidden lines
        is_valid = True
        for f_p in forbidden_polys:
            if poly.intersects(f_p):
                is_valid = False
                break
        is_valid_sg.append(is_valid)
        if is_valid:
            dist = (self.world.goal - b2Vec2(sg)).length
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_sg = sg
    chosen_sg = min_sg
    dir_x = min_sg[0] - agent_pos.x
    dir_y = min_sg[1] - agent_pos.y
    theta = math.atan2(dir_y, dir_x) / (2*math.pi)
    force = 1.0
    last_theta = theta * 2*math.pi
    return np.array([theta, force])


def render():
    screen = env.render()
    self = env.cur_env

    # Draw the forbidden_polys
    # for poly in forbidden_polys:
    #     vertices = list(poly.exterior.coords)
    #     vertices = [self.world.worldToScreen(v) for v in vertices]
    #     cv2.fillPoly(screen, [np.array(vertices)], (255, 0, 0))

    for i in range(len(sg_candidates)):
        sg = sg_candidates[i]
        sg = self.world.worldToScreen(sg)
        if is_valid_sg[i]:
            cv2.circle(screen, sg, 5, (0, 255, 0), -1)
        else:
            cv2.circle(screen, sg, 5, (255, 0, 0), -1)
    
    # Draw the chosen_sg corridor
    # if chosen_sg is not None:
    #     corridor = _gen_rectangle_from_center_line(self, (self.world.agent.agent_rigid_body.position.x, self.world.agent.agent_rigid_body.position.y), chosen_sg)
    #     corridor = [self.world.worldToScreen(v) for v in corridor]
    #     cv2.fillPoly(screen, [np.array(corridor)], (0, 255, 0))

    scene_buffer.PushFrame(screen)
    scene_buffer.Draw()
    cv2.waitKey(12)

config = NavigationEnvConfig(
    world_config= NavigationWorldConfig(
        obstacle_l = [],
        n_rays = 72,
        range_max = 25.0,
        agent_type = 'continuous',
        max_force_length=0.5,
        min_force_length=0.1,
        width=90.0,
        height=80.0,
    ),
    max_steps = 500,
    previous_obs_queue_len = 2,
    reward_scale=10.0
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
env = NavigationMixEnv(config, obs_l_dict)

min_subgoal_pos_dist = 15.0
max_subgoal_pos_dist = 15.0
corridor_width = 10.0
max_subgoal_angle_dist = np.pi/12
max_theta_change = np.pi/3

scene_buffer = CvDrawBuffer(window_name="Simulation", resolution=(1024,1024))

render()
obs, info = env.reset()
acc_reward = 0
success_l = []
while True:
    # Find the direction, take a simple step towards.
    action = find_best_action(env.cur_env)
    # print(min_sg)
    # action = np.array([0.0, 0.0])
    # action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    acc_reward += reward
    render()
    if terminated or truncated:
        success_l.append(info['is_success'])
        obs, info = env.reset()
        print('Reward: ', acc_reward)
        acc_reward = 0
        print('Success rate: ', sum(success_l) / len(success_l))

