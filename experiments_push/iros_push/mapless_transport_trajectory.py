# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.envs.transportation_pose_capsule_conditioned_env import TransportationEnvConfig, TransportationEnv
from research_envs.envs.obstacle_repo_v2 import obstacle_l_dict
from research_envs.envs.object_repo import object_desc_dict
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
from collections import deque


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

Uses the capsule conditioned env. With 2 models. An intermediate for the position and another for pose.
"""

# Env override functions
def adjusted_world_reset(self):
    self.agent_collided = 0
    self.object_collided = 0

    self.obj = self.obj_l[random.randrange(0, len(self.obj_l))]
    # self.obj.obj_rigid_body.position = self.gen_non_overlapping_position(
    #     obj_goal_init_slack)
    # self.obj.obj_rigid_body.angle = random.uniform(0, 2*np.pi)
    self.obj.obj_rigid_body.position = (12.5, 12.5)
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
    self.goal['pos'].x = 140.0
    self.goal['pos'].y = 56.0
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

def get_point_in_line_closest_to_p(p, line_a, line_c):
    # p => b2Vec2
    # line_a, line_b => float
    # Corridor observation
    # Find the closest point to the agent center in the corridor line
    # ax + by +c = 0
    a = line_a
    b = -1.0
    c = line_c
    p_x = (b*(b*p.x - a*p.y) - a*c) / (a*a + b*b)
    p_y = (a*(-b*p.x + a*p.y) - b*c) / (a*a + b*b)
    return b2Vec2(p_x, p_y)

def set_new_goal(self, new_goal={'pos':b2Vec2(0,0), 'angle': 0.0}):    
    self.world.goal = new_goal
    self.step_count = 0
    self.prev_action_queue.clear()
    # self.prev_obs_queue.clear()
    self.last_dist = self.world.object_to_goal_vector().length
    self.last_orient_error = self.world.distToOrientation()/ np.pi

    # Adjust start_obj_pos to lay in the previous corridor
    new_start_obj_pos = get_point_in_line_closest_to_p(
        self.world.obj.obj_rigid_body.position, self.corr_line[0], self.corr_line[1])
    
    self.start_obj_pos = b2Vec2(new_start_obj_pos)
    self.corr_line = [
        (self.world.goal['pos'].y - self.start_obj_pos.y) / (self.world.goal['pos'].x - self.start_obj_pos.x),
        self.start_obj_pos.y - self.start_obj_pos.x * (self.world.goal['pos'].y - self.start_obj_pos.y) / (self.world.goal['pos'].x - self.start_obj_pos.x)
    ]
    return self._gen_observation(), {}

# Trajectory keeping
class TrajectoryMemory:
    """
    Class to keep track of the current trajectory.
    Useful for computing the information gain.
    """
    def __init__(self, min_step_len=0.1, max_step_len=0.5, max_len=10):
        self.max_step_len = max_step_len
        self.min_step_len = min_step_len
        self.trajectory = deque(maxlen=max_len)

    def add(self, pos):
        # pos => np.array([x, y])
        if len(self.trajectory) == 0:
            self.trajectory.append(pos)
        else:
            last_pos = self.trajectory[-1]
            if np.linalg.norm(pos - last_pos) < self.min_step_len:
                return
            # Interpolate between last_pos and pos, add each point
            n_steps = int(np.linalg.norm(pos - last_pos) / self.max_step_len)
            for i in range(n_steps):
                self.trajectory.append(
                    last_pos + (pos - last_pos) * (i / n_steps))
            
            last_pos = self.trajectory[-1]
            if np.linalg.norm(pos - last_pos) >= self.min_step_len:
                self.trajectory.append(pos)

    def get_trajectory_arr(self):
        return np.array(self.trajectory)
    
    def clear(self):
        self.trajectory.clear()

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

def angle_dist(a, b):
    a = a % (2*np.pi)
    if a < 0.0: a += 2*np.pi
    b = b % (2*np.pi)
    if b < 0.0: b += 2*np.pi
    angle_diff = b - a
    if angle_diff > np.pi:
        angle_diff -= 2*np.pi
    elif angle_diff < -np.pi:
        angle_diff += 2*np.pi
    return angle_diff

def compute_corridor_width(subgoal_pos, obstacle_point_arr, ang_arr, laser_angle_range):
    # Checks the laser rays in the direction of the subgoal plus angle
    if len(obstacle_point_arr) == 0: return max_corridor_width

    obj_pos = np.array([
        env.world.obj.obj_rigid_body.position.x, env.world.obj.obj_rigid_body.position.y
    ])
    sg_angle = math.atan2( # returned value is between PI and -PI.
        subgoal_pos[1] - obj_pos[1], 
        subgoal_pos[0] - obj_pos[0])
    
    # LineString from obj to subgoal
    corridor_center = LineString([
        (obj_pos[0], obj_pos[1]),
        (subgoal_pos[0], subgoal_pos[1])
    ])

    min_d = None
    for i in range(obstacle_point_arr.shape[0]):
        if abs(angle_dist(sg_angle, ang_arr[i])) <= laser_angle_range:
            d = corridor_center.distance(Point(obstacle_point_arr[i, 0], obstacle_point_arr[i, 1]))
            if min_d is None or d < min_d:
                min_d = d
    if min_d is None: return max_corridor_width
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
    obstacle_point_arr = np.array([
        (point_l[i].x, point_l[i].y)
        for i in range(len(point_l))
        if type_l[i] == LaserHit.OBSTACLE
    ])
    if obstacle_point_arr.shape[0] == 0:
        ang_arr = np.array([])
    else:
        # In obj centered coordinates
        centered_obstacle_point_arr = obstacle_point_arr - np.array([
            env.world.obj.obj_rigid_body.position.x, env.world.obj.obj_rigid_body.position.y])
        ang_arr = np.arctan2(centered_obstacle_point_arr[:, 1], centered_obstacle_point_arr[:, 0])

    # Compute corridor width for each subgoal
    sg_corridor_width_l = [
        compute_corridor_width(sg_candidate_l[i], obstacle_point_arr, ang_arr, laser_angle_range)
        for i in range(len(sg_candidate_l))
    ]
    viability_score_arr = np.array([
        min(max_corridor_width, sg_corridor_width_l[i]) / corridor_width
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
    d_max = d_score_arr.max() + 1e-6
    d_score_arr = (d_score_arr - d_min) / (d_max - d_min)
    d_score_arr = 1.0 - d_score_arr

    trajectory_arr = trajectory_memory.get_trajectory_arr()
    i_score_arr = np.zeros(len(sg_candidate_l))
    if ((final_goal['pos'] - agent_pos).length > min_dist_to_final_goal_for_info_gain) and (len(trajectory_arr) > 0):
        # Compute distance to each point in the trajectory
        for i in range(len(sg_candidate_l)):
            dist_arr = np.linalg.norm(sg_candidate_l[i] - trajectory_arr, axis=1)
            i_score_arr[i] = sum(dist_arr <= trajectory_close_enough)
        i_score_arr = np.log(1 + i_score_arr)
        i_min = i_score_arr.min()
        i_max = i_score_arr.max() + 1e-6
        i_score_arr = (i_score_arr - i_min) / (i_max - i_min)
        i_score_arr = 1.0 - i_score_arr

    # score_arr = 2.0*viability_score_arr + 0.25*d_score_arr + 0.75*i_score_arr
    score_arr = 2.0*viability_score_arr + 0.9*i_score_arr + 0.1*d_score_arr
    best_sg_idx = np.argmax(score_arr)

    is_valid = sg_corridor_width_l[best_sg_idx] >= corridor_width 
    return sg_candidate_l[best_sg_idx], is_valid


def compute_best_angle(subgoal_pos):
    # If the goal is the final goal, use the angle of the final goal
    if (final_goal['pos'] - b2Vec2(subgoal_pos[0], subgoal_pos[1])).length < subgoal_tolerance['pos']:
        return final_goal['angle']

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
    trajectory_memory.clear()

def adjust_obs(obs):
    return obs#obs[72:]

def adjust_obs_position(obs):
    #np.concatenate((start_obs, goal_obs, obj_obs, caps_obs), dtype=np.float32)
    # Remove the goal angle
    return np.concatenate([obs[:4], obs[5:]])

def _gen_rectangle_from_center_line(start_p, end_p, w):
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

# def render():
#     self = env
#     screen = env.render()

#     # Candidates
#     for i in range(len(sg_candidate_l)):
#         sg = sg_candidate_l[i]
#         sg = self.world.worldToScreen(sg)
#         if sg_corridor_width_l[i] < corridor_width:
#             cv2.circle(screen, sg, 5, (255, 0, 0), -1)
#         else:
#             cv2.circle(screen, sg, 5, (0, 255, 0), -1)

#     # Draw trajectory as little red arrows
#     trajectory_arr = trajectory_memory.get_trajectory_arr()
#     for i in range(len(trajectory_arr)):
#         if i == 0: continue
#         start = self.world.worldToScreen(trajectory_arr[i-1])
#         end = self.world.worldToScreen(trajectory_arr[i])
#         screen = cv2.arrowedLine(screen, start, end, color=(0, 0, 255), thickness=6)

#     # Draw final goal in yellow
#     if final_goal is not None:
#         self.world.obj.DrawInPose(
#             final_goal['pos'], final_goal['angle'], self.world.pixels_per_meter, screen, (0, 255, 255), -1)
#         self.world.drawArrow(screen, final_goal['pos'], final_goal['angle'], 10, (0, 255, 255))

#     # Draw corridor
#     if min_sg is not None:
#         corridor = _gen_rectangle_from_center_line((self.start_obj_pos.x, self.start_obj_pos.y), min_sg, corridor_width)
#         corridor = [self.world.worldToScreen(v) for v in corridor]
#         cv2.polylines(screen, [np.array(corridor)], isClosed=True, color=(0, 255, 0), thickness=4)
#         sg_radius = int(corridor_width * self.world.pixels_per_meter)
#         cv2.circle(screen, self.world.worldToScreen(min_sg), sg_radius, (0, 255, 0), thickness=4)

#     frame_l.append(screen)
#     scene_buffer.PushFrame(screen)
#     scene_buffer.Draw()
#     cv2.waitKey(50)

def DrawFirstVerticeInPose(self, world_pos, angle, pixels_per_meter, image, color, thickness, radius=5):
    transform_matrix = b2Transform()
    transform_matrix.SetIdentity()
    transform_matrix.Set(world_pos, angle)
    body = self.obj_rigid_body
    first_v = (transform_matrix * body.fixtures[0].shape.vertices[0])
    cv2.circle(image, self.worldToScreen(first_v, pixels_per_meter), radius, color, thickness)

def drawPretty(self):
    # clear previous buffer
    screen = 255 * np.ones(shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8)
    # screen[:,:,0] = 244
    # screen[:,:,1] = 233
    # screen[:,:,2] = 239
    # Draw obstacles
    for obs in self.obstacle_l:
        # obs.Draw(self.pixels_per_meter, screen, (109, 77, 72), -1)
        obs.Draw(self.pixels_per_meter, screen, (128, 128, 128), -1)
    # Draw the object
    self.obj.Draw(self.pixels_per_meter, screen, (82, 99, 238), -1)
    DrawFirstVerticeInPose(
            self.obj, self.obj.GetPositionAsList(), self.obj.obj_rigid_body.angle, 
            self.pixels_per_meter, screen, (36, 21, 0), -1, 10)

    # Draw agent
    screen_pos = self.worldToScreen(self.agent.GetPositionAsList())
    cv2.circle(screen, screen_pos, int(self.agent.agent_radius*self.pixels_per_meter), (214, 167, 63), -1)
    return screen

def render():
    self = env
    screen = drawPretty(env.world)

    # Draw trajectory as little red arrows
    trajectory_arr = trajectory_memory.get_trajectory_arr()
    for i in range(len(trajectory_arr)):
        if i == 0: continue
        start = self.world.worldToScreen(trajectory_arr[i-1])
        end = self.world.worldToScreen(trajectory_arr[i])
        screen = cv2.line(screen, start, end, color=(36, 21, 0), thickness=6)
        # screen = cv2.arrowedLine(screen, start, end, color=(36, 21, 0), thickness=6)

    # Draw final goal in green
    if final_goal is not None:
        self.world.obj.DrawInPose(
            final_goal['pos'], final_goal['angle'], self.world.pixels_per_meter, screen, (151, 207, 100), -1)
        DrawFirstVerticeInPose(
            env.world.obj, final_goal['pos'], final_goal['angle'], 
            env.world.pixels_per_meter, screen, (36, 21, 0), -1, 10)

    frame_l.append(screen)
    scene_buffer.PushFrame(screen)
    scene_buffer.Draw()
    cv2.waitKey(50)


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


# MAIN    
###################### OBJECT ID ######################
obj_id = int(sys.argv[1])
exp_name = 'obj_' + str(obj_id)
map_obs = obstacle_l_dict[sys.argv[2]]
corridor_width = float(sys.argv[3])
corridor_width_for_robot = float(sys.argv[4])

# Parameters
# corridor_width = 8.5
# corridor_width_for_robot = 10.0
max_corridor_width = corridor_width
obj_goal_init_slack = corridor_width #* 1.1
# Only evaluates laser rays in the direction of the candidate plus/minus this angle
laser_angle_range = np.pi/2

# If dist to final goal > pose_model_dist, use the position model
pose_model_dist = 100.0

macro_env_max_steps = 1000
macro_env_steps = 0
final_goal = None

candidate_dist_range = (5.0, 10.0)
n_candidates = 200

max_angle_diff = np.pi/6
subgoal_tolerance = {'pos':4, 'angle':np.pi}
micro_env_max_steps = 30
micro_env_steps = 0
sg_is_valid = False
min_sg = None

# Information gain
trajectory_memory = TrajectoryMemory(min_step_len=2.0, max_step_len=2.0, max_len=300)
trajectory_close_enough = 10.0
min_dist_to_final_goal_for_info_gain = 10.0

sg_candidate_l = []
sg_score_l = []
sg_corridor_width_l = []

# If the object gets stuck, move towards it.
stuck_cnt = 5
obj_pos_deque = deque(maxlen=stuck_cnt)

config = TransportationEnvConfig(
    world_config= TransportationWorldConfig(
        obstacle_l = map_obs['obstacles'],
        object_l=[object_desc_dict[obj_id]],
        n_rays = 72,
        range_max = 25.0,
        agent_type = 'continuous',
        max_force_length=5.0,
        min_force_length=0.1,
        width=map_obs['width'],
        height=map_obs['height'],
        goal_tolerance={'pos':2, 'angle':np.pi/18},
        max_obj_dist=10.0
    ),
    max_steps = 100,
    previous_obs_queue_len = 0,
    reward_scale=10.0,
    corridor_width_range = (corridor_width_for_robot, corridor_width_for_robot)
)

env = TransportationEnv(config)
# Overrides
env.world.reset = types.MethodType(adjusted_world_reset, env.world)
env._check_death = types.MethodType(adjusted_check_death, env)
env._check_success = types.MethodType(adjusted_check_success, env)

# Load agent
model = SAC.load('model_ckp/'+exp_name+'/best_model')

# Rendering
scene_buffer = CvDrawBuffer(window_name="Simulation", resolution=(1024,1024))
frame_l = []
video_counter = 0

# Trajectory draw
robot_trajectory = []
obj_trajectory = []

render_bool = False

if render_bool: render()
obs, info = env.reset()
reset_macro_env()
done = False
while not done:
    # Record trajectory
    obj_trajectory.append({
        'pos': b2Vec2(env.world.obj.obj_rigid_body.position),
        'angle': env.world.obj.obj_rigid_body.angle
    })
    robot_trajectory.append({
        'pos': b2Vec2(env.world.agent.agent_rigid_body.position)
    })

    # Update subgoal
    if check_subgoal_sucess(env): micro_env_steps = 0
    if not sg_is_valid: micro_env_steps  = 0
    if micro_env_steps % micro_env_max_steps == 0:
        obj_pos = np.array(
            [env.world.obj.obj_rigid_body.position.x, env.world.obj.obj_rigid_body.position.y]
        )
        trajectory_memory.add(obj_pos)
        min_sg, sg_is_valid = find_best_subgoal()
        set_new_goal(env, new_goal={'pos':b2Vec2(min_sg), 'angle': compute_best_angle(min_sg)})
        micro_env_steps = 0
    micro_env_steps += 1

    # Take action
    # If the object is stuck, move towards it
    obj_stuck = False
    obj_pos_deque.append(np.array(
        [env.world.obj.obj_rigid_body.position.x, env.world.obj.obj_rigid_body.position.y]
    ))
    if len(obj_pos_deque) == stuck_cnt:
        if np.linalg.norm(obj_pos_deque[0] - obj_pos_deque[-1]) < 0.1:
            obj_stuck = True

    if obj_stuck:
        agent_to_obj = env.world.agent_to_object_vector()
        angle = np.arctan2(agent_to_obj[1], agent_to_obj[0])
        # print('Stuck')
        if angle < 0: angle += 2*np.pi
        action = np.array([angle / (2*np.pi), 0.5])
    else:
        # print('Pose Model')
        obs = adjust_obs(obs)
        action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if render_bool: render()

    macro_env_steps += 1
    if macro_env_steps >= macro_env_max_steps:
        truncated = True

    done = terminated or truncated


def draw_obj_and_robot(screen, obj_pose, robot_pose):
    # Object
    env.world.obj.DrawInPose(
        obj_pose['pos'], obj_pose['angle'], env.world.pixels_per_meter, screen, (82, 99, 238), -1)
    DrawFirstVerticeInPose(
        env.world.obj, obj_pose['pos'], obj_pose['angle'], env.world.pixels_per_meter, screen, (36, 21, 0), -1, 10)
    # Robot
    cv2.circle(
        screen, 
        env.world.worldToScreen(robot_pose['pos']), 
        int(env.world.agent.agent_radius*env.world.pixels_per_meter), 
        color=(214, 167, 63), thickness=-1)



# Generate a picture of the trajectory
screen = 255 * np.ones(shape=(
    env.world.screen_height, env.world.screen_width, 3), dtype=np.uint8)
    
# Draw the obstacles
for obs in env.world.obstacle_l:
    obs.Draw(env.world.pixels_per_meter, screen, (128, 128, 128), -1)

# Draw the first
draw_obj_and_robot(screen, obj_trajectory[0], robot_trajectory[0])

# Draw parts of the trajectory
dist_gap = 20.0
last_pos = obj_trajectory[0]['pos']
for i in range(1, len(obj_trajectory)):
    if (obj_trajectory[i]['pos'] - last_pos).length > dist_gap:
        draw_obj_and_robot(screen, obj_trajectory[i], robot_trajectory[i])
        last_pos = obj_trajectory[i]['pos']

# Draw the goal
env.world.obj.DrawInPose(
    final_goal['pos'], final_goal['angle'], env.world.pixels_per_meter, screen, (151, 207, 100), -1)
DrawFirstVerticeInPose(
    env.world.obj, final_goal['pos'], final_goal['angle'], env.world.pixels_per_meter, screen, (36, 21, 0), -1, 10)

# Draw the object trajectory as a curve
c = (36, 21, 0)
thickness = 4
for i in range(len(obj_trajectory)):
    if i == 0 or i == len(obj_trajectory)-1: continue
    cv2.line(
        screen, 
        env.world.worldToScreen(obj_trajectory[i-1]['pos']), 
        env.world.worldToScreen(obj_trajectory[i]['pos']), 
        color=c, thickness=thickness)
cv2.line(
    screen, 
    env.world.worldToScreen(obj_trajectory[-2]['pos']), 
    env.world.worldToScreen(obj_trajectory[-1]['pos']), 
    color=c, thickness=thickness)
cv2.line(
    screen, 
    env.world.worldToScreen(obj_trajectory[-1]['pos']), 
    env.world.worldToScreen(final_goal['pos']), 
    color=c, thickness=thickness)

cv2.imwrite('mapless_trajectory.png', screen)