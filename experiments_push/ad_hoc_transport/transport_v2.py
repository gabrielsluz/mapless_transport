# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.envs.transportation_pose_corridor_env import TransportationEnvConfig, TransportationEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from research_envs.b2PushWorld.TransportationPoseWorld import TransportationWorldConfig, LaserHit
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

import cv2

from shapely import LineString, Polygon, Point
from Box2D import b2Vec2, b2Transform

from stable_baselines3 import SAC

import random
import math
import numpy as np
import types


"""
Ad Hoc Navigation

Laser: detects obstacle points.

Subgoal generation: sample (x, y) in a distance (ro) and direction (theta) ranges.
If final_goal inside range => append to candidates.

For each candidate:
- Compute maximum corridor by using the obstacle points. => min dist to line segment (cur_pos, cand)
- Based on the distance to candidate, compute a nice orientation subgoal.
- Compute distance to final goal.
- Compute information gain weighted by how many times in the same spot.
Compute final score: S = 2*V + 1/2*(D + I)
Pick the highest score.

TODO:
Organize this crap.
Env leva até o final_goal.
Precisa dar override em funções do env.
Overrides:
- check_death
- check_success
- world reset

Stop using mix env

- Colocar um max steps
"""

# Env override functions
def adjusted_world_reset(self):
    self.agent_collided = 0
    self.object_collided = 0

    self.obj = self.obj_l[random.randrange(0, len(self.obj_l))]
    self.obj.obj_rigid_body.position = self.gen_non_overlapping_position(
        obj_goal_init_slack)
    self.obj.obj_rigid_body.angle = random.uniform(0, 2*np.pi)

    x_lim = [
        self.obj.obj_rigid_body.position.x - self.max_obj_dist*0.7071,
        self.obj.obj_rigid_body.position.x + self.max_obj_dist*0.7071
    ]
    y_lim = [
        self.obj.obj_rigid_body.position.y - self.max_obj_dist*0.7071,
        self.obj.obj_rigid_body.position.y + self.max_obj_dist*0.7071
    ]
    self.agent.agent_rigid_body.position = self.gen_non_overlapping_position_in_limit(
        self.agent.agent_radius*1.2, x_lim, y_lim)

    sampled_pos = self.gen_non_overlapping_position(obj_goal_init_slack)
    self.goal['pos'].x = sampled_pos[0]
    self.goal['pos'].y = sampled_pos[1]
    self.goal['angle'] = random.uniform(0, 2*np.pi)

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

# Goal setting
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

# Ad hoc navigation functions
def _gen_subgoal_candidate(env, dist_range=(0.0, 10.0), angle_range=(0, 2*np.pi)):
    # Gen subgoals in a range radius from the objects position
    rand_dist = random.uniform(dist_range[0], dist_range[1])
    rand_rad = random.uniform(angle_range[0], angle_range[1])
    subgoal_pos = [
        env.world.obj.obj_rigid_body.position.x + rand_dist * np.cos(rand_rad),
        env.world.obj.obj_rigid_body.position.y + rand_dist * np.sin(rand_rad)
    ]
    return subgoal_pos

def compute_corridor_width(subgoal_pos, obstacle_point_l):
    if len(obstacle_point_l) == 0: return max_corridor_width
    # LineString from obj to subgoal
    corridor_center = LineString([
        (env.world.obj.obj_rigid_body.position.x, env.world.obj.obj_rigid_body.position.y),
        (subgoal_pos[0], subgoal_pos[1])
    ])
    min_d = None
    for p in obstacle_point_l:
        d = corridor_center.distance(Point(p.x, p.y))
        if min_d is None or d < min_d:
            min_d = d
    return min_d

def find_best_subgoal():
    global sg_candidate_l, sg_score_l, sg_corridor_width_l
    # Generates candidates, score them and picks the highest score.

    sg_candidate_l = [
        _gen_subgoal_candidate(env, candidate_dist_range, (0, 2*np.pi)) 
        for _ in range(n_candidates)
    ]

    agent_pos = env.world.obj.obj_rigid_body.position
    if (final_goal['pos'] - agent_pos).length <= candidate_dist_range[1]:
        sg_candidate_l.append([final_goal['pos'].x, final_goal['pos'].y])

    # Obstacle detection
    _, type_l, point_l = env.world.get_laser_readings()
    # Get the points with obstacles
    obstacle_point_l = [
        Point(point_l[i].x, point_l[i].y)
        for i in range(len(point_l)) 
        if type_l[i] == LaserHit.OBSTACLE]
    # Compute corridor width for each subgoal
    sg_corridor_width_l = [
        compute_corridor_width(sg_candidate_l[i], obstacle_point_l)
        for i in range(len(sg_candidate_l))
    ]
    viability_score_arr = np.array([
        1.0 
        if sg_corridor_width_l[i] >= corridor_width 
        else 0.5*(sg_corridor_width_l[i]/corridor_width)
        for i in range(len(sg_candidate_l))
    ])

    # Compute distance to final goal
    d_score_arr = np.array([
        (final_goal['pos'] - b2Vec2(sg_candidate_l[i][0], sg_candidate_l[i][1])).length
        for i in range(len(sg_candidate_l))
    ])

    d_min = d_score_arr.min()
    d_max = d_score_arr.max()
    d_score_arr = (d_score_arr - d_min) / (d_max - d_min)
    d_score_arr = 1.0 - d_score_arr

    score_arr = 2.0*viability_score_arr + d_score_arr
    best_sg_idx = np.argmax(score_arr)

    is_valid = sg_corridor_width_l[best_sg_idx] >= corridor_width 
    return sg_candidate_l[best_sg_idx], is_valid


def compute_best_angle():
    # We start from
    cur_angle = env.world.obj.obj_rigid_body.angle % (2*np.pi)
    # Want to get to:
    goal_angle = final_goal['angle'] % (2*np.pi)
    # Compute the angle difference
    angle_diff = goal_angle - cur_angle
    if angle_diff > np.pi:
        angle_diff -= 2*np.pi
    elif angle_diff < -np.pi:
        angle_diff += 2*np.pi

    if abs(angle_diff) < max_angle_diff:
        return goal_angle
    else:
        # Compute the best angle
        if angle_diff > 0.0:
            return cur_angle + max_angle_diff
        else:
            return cur_angle - max_angle_diff





def reset_macro_env():
    # Run after env.reset()
    global final_goal, macro_env_steps, micro_env_steps, sg_is_valid
    final_goal = {'pos':b2Vec2(env.world.goal['pos']), 'angle': env.world.goal['angle']}
    macro_env_steps = 0
    micro_env_steps = 0
    sg_is_valid = False

def adjust_obs(obs):
    return obs[72:]

def render():
    self = env
    screen = env.render()

    # Candidates
    for i in range(len(sg_candidate_l)):
        sg = sg_candidate_l[i]
        sg = self.world.worldToScreen(sg)
        if sg_corridor_width_l[i] < corridor_width:
            cv2.circle(screen, sg, 5, (255, 0, 0), -1)
        else:
            cv2.circle(screen, sg, 5, (0, 255, 0), -1)
    
    # Draw best subgoal in red
    if min_sg is not None:
        screen_pos = self.world.worldToScreen(min_sg)
        cv2.circle(screen, screen_pos, 10, (0, 0, 255), -1)

    # Draw final goal in yellow
    if final_goal is not None:
        self.world.obj.DrawInPose(
            final_goal['pos'], final_goal['angle'], self.world.pixels_per_meter, screen, (0, 255, 255), -1)
        self.world.drawArrow(screen, final_goal['pos'], final_goal['angle'], 10, (0, 255, 255))

    scene_buffer.PushFrame(screen)
    scene_buffer.Draw()
    cv2.waitKey(25)


# MAIN
# Parameters
corridor_width = 10.0
max_corridor_width = 10.0
obj_goal_init_slack = 10.0

macro_env_max_steps = 500
macro_env_steps = 0
final_goal = None

candidate_dist_range = (5.0, 15.0)
n_candidates = 100

max_angle_diff = np.pi/6
subgoal_tolerance = {'pos':3, 'angle':np.pi/4}
micro_env_max_steps = 10
micro_env_steps = 0
sg_is_valid = False
min_sg = None

sg_candidate_l = []
sg_score_l = []
sg_corridor_width_l = []

object_desc = {'name': 'Rectangle', 'height': 10.0, 'width': 5.0}
print(object_desc)

config = TransportationEnvConfig(
    world_config= TransportationWorldConfig(
        obstacle_l = obstacle_l_dict['big_sparse_2'],
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

env = TransportationEnv(config)
# Overrides
env.world.reset = types.MethodType(adjusted_world_reset, env.world)
env._check_death = types.MethodType(adjusted_check_death, env)
env._check_success = types.MethodType(adjusted_check_success, env)

# Load agent
model = SAC.load('model_ckp/progress_sac_rectangle_tolerance_pi18_pos_tol_2_reward_scale_10_corridor_full_death_width_10')

# Rendering
scene_buffer = CvDrawBuffer(window_name="Simulation", resolution=(1024,1024))

render()
obs, info = env.reset()
reset_macro_env()
acc_reward = 0
success_l = []
while True:
    # Update subgoal
    if check_subgoal_sucess(env): micro_env_steps = 0
    if not sg_is_valid: micro_env_steps  = 0
    if micro_env_steps % micro_env_max_steps == 0:
        min_sg, sg_is_valid = find_best_subgoal()
        set_new_goal(env, new_goal={'pos':b2Vec2(min_sg), 'angle': compute_best_angle()})
        micro_env_steps = 0
    micro_env_steps += 1

    # Take action
    obs = adjust_obs(obs)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    acc_reward += reward
    render()

    macro_env_steps += 1
    if macro_env_steps >= macro_env_max_steps:
        truncated = True

    if terminated or truncated:
        success_l.append(info['is_success'])
        obs, info = env.reset()
        reset_macro_env()

        print('Reward: ', acc_reward)
        acc_reward = 0
        print('Success rate: ', sum(success_l) / len(success_l))

