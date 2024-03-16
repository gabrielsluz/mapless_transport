"""
Runs an episode and records the objects trajectory.
"""
# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.b2PushWorld.TransportationPoseWorld import TransportationWorldConfig
from research_envs.envs.transportation_pose_capsule_conditioned_env import TransportationEnvConfig, TransportationEnv
from research_envs.envs.object_repo import object_desc_dict
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

from Box2D import b2Vec2, b2Transform
import copy
import random

import cv2
import numpy as np
from stable_baselines3 import SAC

def DrawFirstVerticeInPose(self, world_pos, angle, pixels_per_meter, image, color, thickness, radius=5):
    transform_matrix = b2Transform()
    transform_matrix.SetIdentity()
    transform_matrix.Set(world_pos, angle)
    body = self.obj_rigid_body
    first_v = (transform_matrix * body.fixtures[0].shape.vertices[0])
    cv2.circle(image, self.worldToScreen(first_v, pixels_per_meter), radius, color, thickness)

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
    cv2.waitKey(10)

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
        goal_tolerance={'pos':1, 'angle':np.pi/18},
        max_obj_dist=8.0
    ),
    max_steps = 500,
    previous_obs_queue_len = 0,
    reward_scale=10.0,
    corridor_width_range = (8.0, 12.0)
)

env = TransportationEnv(config)
model = SAC.load('model_ckp/'+exp_name+'/best_model')
print(model.policy)

scene_buffer = CvDrawBuffer(window_name="Simulation", resolution=(1024,1024))

# Manually reset the env to control the initial position of the object
# World reset:
env.world.agent_collided = 0
env.world.object_collided = 0

env.world.goal['pos'].x = 55
env.world.goal['pos'].y = 25
env.world.goal['angle'] = random.uniform(0, 2*np.pi)#np.deg2rad(40)

env.world.obj = env.world.obj_l[0]
env.world.obj.obj_rigid_body.position = (15, 25)
env.world.obj.obj_rigid_body.angle = random.uniform(0, 2*np.pi)#np.deg2rad(200)

# env.world.agent.agent_rigid_body.position = (10, 25)
x_lim = [
    env.world.obj.obj_rigid_body.position.x - env.world.max_obj_dist*0.7071,
    env.world.obj.obj_rigid_body.position.x + env.world.max_obj_dist*0.7071
]
y_lim = [
    env.world.obj.obj_rigid_body.position.y - env.world.max_obj_dist*0.7071,
    env.world.obj.obj_rigid_body.position.y + env.world.max_obj_dist*0.7071
]
# env.world.agent.agent_rigid_body.position = env.world.gen_non_overlapping_position_in_limit(
#     env.world.agent.agent_radius*1.2, x_lim, y_lim)
theta = random.uniform(0, 2*np.pi)
v = b2Vec2(np.cos(theta), np.sin(theta))
v = v * (env.world.obj.obj_radius + 1.1*env.world.agent.agent_radius)
env.world.agent.agent_rigid_body.position = env.world.obj.obj_rigid_body.position + v

env.step_count = 0
env.corridor_width = 10.0
env._update_corridor_variables()

obs = env._gen_observation()

trajectory = []
trajectory_robot = []


# render()
acc_reward = 0
success_l = []
while True:
    trajectory.append({
        'pos': b2Vec2(env.world.obj.obj_rigid_body.position),
        'angle': env.world.obj.obj_rigid_body.angle
    })

    trajectory_robot.append({
        'pos': b2Vec2(env.world.agent.agent_rigid_body.position)
    })

    print(env.world.obj.obj_rigid_body.position, env.world.obj.obj_rigid_body.angle)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    acc_reward += reward
    # render()
    if terminated or truncated:
        success_l.append(info['is_success'])
        print()
        print('Reward: ', acc_reward)
        acc_reward = 0
        print('Success rate: ', sum(success_l) / len(success_l))
        break

trajectory.append({
    'pos': b2Vec2(env.world.obj.obj_rigid_body.position),
    'angle': env.world.obj.obj_rigid_body.angle
})

trajectory_robot.append({
    'pos': b2Vec2(env.world.agent.agent_rigid_body.position)
})

# SAVE THE IMAGE:
screen = 255 * np.ones(shape=(
    env.world.screen_height, int(env.world.screen_width*1.5), 3), dtype=np.uint8)

# Draw the capsule
capsule_line = env.capsule_line
capsule = capsule_line.buffer(env.corridor_width)
# Convert the Shapely object to a list of points
points = np.array([list(point) for point in capsule.exterior.coords])
# Scale the points to match the pixels_per_meter ratio
points = (points * env.world.pixels_per_meter).astype(int)
# Reshape the points to the format expected by cv2.polylines
points = points.reshape((-1, 1, 2))
# Draw the capsule
cv2.polylines(screen, [points], isClosed=True, color=(36, 21, 0), thickness=3)

# Draw the goal
env.world.obj.DrawInPose(
    env.world.goal['pos'], env.world.goal['angle'], env.world.pixels_per_meter, screen, (151, 207, 100), -1)
DrawFirstVerticeInPose(
    env.world.obj, env.world.goal['pos'], env.world.goal['angle'], env.world.pixels_per_meter, screen, (36, 21, 0), -1, 10)

# Draw the first
env.world.obj.DrawInPose(
    trajectory[0]['pos'], trajectory[0]['angle'], env.world.pixels_per_meter, screen, (82, 99, 238), -1)
DrawFirstVerticeInPose(
    env.world.obj, trajectory[0]['pos'], trajectory[0]['angle'], env.world.pixels_per_meter, screen, (36, 21, 0), -1, 10)

cv2.circle(
    screen, 
    env.world.worldToScreen(trajectory_robot[0]['pos']), 
    int(env.world.agent.agent_radius*env.world.pixels_per_meter), 
    color=(214, 167, 63), thickness=-1)

# Draw the middle
# Find the middle in terms of distance:
aux = (env.world.goal['pos'].x - trajectory_robot[0]['pos'].x) / 3
first_quarter = trajectory_robot[0]['pos'].x + aux
last_quarter = trajectory_robot[0]['pos'].x + 2*aux

# middle_pos = (env.world.goal['pos'].x + trajectory_robot[0]['pos'].x) / 2
# # first_quarter = (trajectory_robot[0]['pos'] + trajectory_robot[-1]['pos']) / 4
# # last_quarter = (trajectory_robot[0]['pos'] + trajectory_robot[-1]['pos']) * 3 / 4
# first_quarter = (trajectory_robot[0]['pos'].x + middle_pos) / 2
# last_quarter = (trajectory_robot[-1]['pos'].x + middle_pos) / 2

print('Start: ', trajectory_robot[0]['pos'])
print('First quarter: ', first_quarter)
print('Last quarter: ', last_quarter)
print('Goal: ', env.world.goal['pos'])
closest_i = 0
closest_dist = 100000
for i in range(len(trajectory)):
    #dist = np.linalg.norm(np.array(trajectory[i]['pos']) - np.array(first_quarter))
    dist = abs(trajectory[i]['pos'].x - first_quarter)
    if dist < closest_dist:
        closest_dist = dist
        closest_i = i
i = closest_i

print('I =', i)
env.world.obj.DrawInPose(
    trajectory[i]['pos'], trajectory[i]['angle'], env.world.pixels_per_meter, screen, (82, 99, 238), -1)
DrawFirstVerticeInPose(
    env.world.obj, trajectory[i]['pos'], trajectory[i]['angle'], env.world.pixels_per_meter, screen, (36, 21, 0), -1, 10)

cv2.circle(
    screen, 
    env.world.worldToScreen(trajectory_robot[i]['pos']), 
    int(env.world.agent.agent_radius*env.world.pixels_per_meter), 
    color=(214, 167, 63), thickness=-1)

# Last quarter
closest_i = 0
closest_dist = 100000
for i in range(len(trajectory)):
    # dist = np.linalg.norm(np.array(trajectory[i]['pos']) - np.array(last_quarter))
    dist = abs(trajectory[i]['pos'].x - last_quarter)
    if dist < closest_dist:
        closest_dist = dist
        closest_i = i
i = closest_i
print('I =', i)
env.world.obj.DrawInPose(
    trajectory[i]['pos'], trajectory[i]['angle'], env.world.pixels_per_meter, screen, (82, 99, 238), -1)
DrawFirstVerticeInPose(
    env.world.obj, trajectory[i]['pos'], trajectory[i]['angle'], env.world.pixels_per_meter, screen, (36, 21, 0), -1, 10)

cv2.circle(
    screen, 
    env.world.worldToScreen(trajectory_robot[i]['pos']), 
    int(env.world.agent.agent_radius*env.world.pixels_per_meter), 
    color=(214, 167, 63), thickness=-1)

# Draw the last
env.world.obj.DrawInPose(
    trajectory[-1]['pos'], trajectory[-1]['angle'], env.world.pixels_per_meter, screen, (82, 99, 238), -1)
DrawFirstVerticeInPose(
    env.world.obj, trajectory[-1]['pos'], trajectory[-1]['angle'], env.world.pixels_per_meter, screen, (36, 21, 0), -1, 10)

cv2.circle(
    screen, 
    env.world.worldToScreen(trajectory_robot[-1]['pos']), 
    int(env.world.agent.agent_radius*env.world.pixels_per_meter), 
    color=(214, 167, 63), thickness=-1)

# Draw the trajectory as a curve
c = (36, 21, 0)
thickness = 2
for i in range(len(trajectory)):
    if i == 0 or i == len(trajectory)-1: continue
    cv2.line(
        screen, 
        env.world.worldToScreen(trajectory[i-1]['pos']), 
        env.world.worldToScreen(trajectory[i]['pos']), 
        color=c, thickness=thickness)
cv2.line(
    screen, 
    env.world.worldToScreen(trajectory[-2]['pos']), 
    env.world.worldToScreen(trajectory[-1]['pos']), 
    color=c, thickness=thickness)
cv2.line(
    screen, 
    env.world.worldToScreen(trajectory[-1]['pos']), 
    env.world.worldToScreen(env.world.goal['pos']), 
    color=c, thickness=thickness)
print(screen.shape)

def DrawInPose(self, world_pos, angle, pixels_per_meter, image, color, thickness):
    transform_matrix = b2Transform()
    transform_matrix.SetIdentity()
    transform_matrix.Set(world_pos, angle)
    body = self.obj_rigid_body
    for f_i in range(len(body.fixtures)):
        vertices = [(transform_matrix * v) for v in body.fixtures[f_i].shape.vertices]
        vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
        cv2.polylines(image, [np.array(vertices)], isClosed=True, color=color, thickness=thickness)

# Center the image on the capsule
s = 0.5
w = 10
start_pos = env.start_obj_pos
end_pos = env.world.goal['pos']
x_rng = [
    start_pos.x - w - s,
    end_pos.x + w + s
]
x_rng_img = [
    env.world.worldToScreen((x_rng[0], 0))[0],
    env.world.worldToScreen((x_rng[1], 0))[0]
]

y_rng = [
    start_pos.y - w - s,
    end_pos.y + w + s
]
y_rng_img = [
    env.world.worldToScreen((0, y_rng[0]))[1],
    env.world.worldToScreen((0, y_rng[1]))[1]
]

screen = screen[y_rng_img[0]:y_rng_img[1], x_rng_img[0]:x_rng_img[1]]
# Save img in file img.png
cv2.imwrite('capsule_videos/img_{}.png'.format(exp_name), screen)


# SAVE THE VIDEO

# Write the frames to a video
# Fixed Capsule
# What changes is the object position and angle
# And the robot position

frame_l = []

# trajectory
# trajectory_robot
for i in range(len(trajectory)):
    # Draw the capsule
    screen = 255 * np.ones(shape=(
        env.world.screen_height, int(env.world.screen_width*1.5), 3), dtype=np.uint8)

    # Draw the capsule
    capsule_line = env.capsule_line
    capsule = capsule_line.buffer(env.corridor_width)
    # Convert the Shapely object to a list of points
    points = np.array([list(point) for point in capsule.exterior.coords])
    # Scale the points to match the pixels_per_meter ratio
    points = (points * env.world.pixels_per_meter).astype(int)
    # Reshape the points to the format expected by cv2.polylines
    points = points.reshape((-1, 1, 2))
    # Draw the capsule
    cv2.polylines(screen, [points], isClosed=True, color=(36, 21, 0), thickness=3)

    # Draw the goal
    env.world.obj.DrawInPose(
        env.world.goal['pos'], env.world.goal['angle'], env.world.pixels_per_meter, screen, (151, 207, 100), -1)
    DrawFirstVerticeInPose(
        env.world.obj, env.world.goal['pos'], env.world.goal['angle'], env.world.pixels_per_meter, screen, (36, 21, 0), -1, 10)

    # Draw the current object
    env.world.obj.DrawInPose(
        trajectory[i]['pos'], trajectory[i]['angle'], env.world.pixels_per_meter, screen, (82, 99, 238), -1)
    DrawFirstVerticeInPose(
        env.world.obj, trajectory[i]['pos'], trajectory[i]['angle'], env.world.pixels_per_meter, screen, (36, 21, 0), -1, 10)

    cv2.circle(
        screen, 
        env.world.worldToScreen(trajectory_robot[i]['pos']), 
        int(env.world.agent.agent_radius*env.world.pixels_per_meter), 
        color=(214, 167, 63), thickness=-1)
    
    # Draw the trajectory as a curve up to i
    c = (36, 21, 0)
    thickness = 2
    for j in range(i):
        if j == 0 or j == len(trajectory)-1: continue
        cv2.line(
            screen, 
            env.world.worldToScreen(trajectory[j-1]['pos']), 
            env.world.worldToScreen(trajectory[j]['pos']), 
            color=c, thickness=thickness)
    

    # Center the image on the capsule
    # s = 2
    # w = 12
    start_pos = env.start_obj_pos
    end_pos = env.world.goal['pos']
    x_rng = [
        start_pos.x - w - s,
        end_pos.x + w + s
    ]
    x_rng_img = [
        env.world.worldToScreen((x_rng[0], 0))[0],
        env.world.worldToScreen((x_rng[1], 0))[0]
    ]

    y_rng = [
        start_pos.y - w - s,
        end_pos.y + w + s
    ]
    y_rng_img = [
        env.world.worldToScreen((0, y_rng[0]))[1],
        env.world.worldToScreen((0, y_rng[1]))[1]
    ]

    screen = screen[y_rng_img[0]:y_rng_img[1], x_rng_img[0]:x_rng_img[1]]

    # # screen = np.transpose(screen, (1, 0, 2))
    # aux_screen = np.zeros(shape=screen.shape, dtype=np.uint8)
    # aux_screen[:,:,0] = screen[:,:,2]
    # aux_screen[:,:,1] = screen[:,:,1]
    # aux_screen[:,:,2] = screen[:,:,0]
    # Increase the brightness by 50
    screen = cv2.convertScaleAbs(screen, alpha=1.0, beta=0)
    frame_l.append(screen)

# Save the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
f_name = 'capsule_videos/video_' + exp_name + '.mp4'
out = cv2.VideoWriter(f_name, fourcc, 10, (frame_l[0].shape[1], frame_l[0].shape[0]))
for frame in frame_l:
    out.write(frame)
out.release()



# def save_video(frame_l):
#     global video_counter
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     f_name = 'videos/video_' + exp_name + '.mp4'
#     out = cv2.VideoWriter(f_name, fourcc, 20, (frame_l[0].shape[1], frame_l[0].shape[0]))
#     for frame in frame_l:
#         frame = frame.astype(np.uint8)
#         out.write(frame)
#     out.release()