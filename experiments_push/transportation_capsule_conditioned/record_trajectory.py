"""
Runs an episode and records the objects trajectory.
"""
# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.b2PushWorld.TransportationPoseWorld import TransportationWorldConfig
from research_envs.envs.transportation_pose_capsule_conditioned_env import TransportationEnvConfig, TransportationEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

from Box2D import b2Vec2, b2Transform
import copy

import cv2
import numpy as np
from stable_baselines3 import SAC

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

# object_desc = {'name': 'Rectangle', 'height': 10.0, 'width': 5.0}
object_desc = {'name': 'Polygon', 'vertices':[[-4, -2], [4, -2], [0, 6]]}
# object_desc = {
#     'name': 'MultiPolygons', 'poly_vertices_l':[
#     [[0, 0], [0, 4], [2, 4], [4, 2], [4, 0]],
#     [[0, 0], [0, 2], [-6, 2], [-6, 0]],
#     [[0, 0], [-4, -6], [0, -4]],

#     [[0, 0], [0, -4], [4, -4]],
#     [[0, 0], [2, -2], [4, 0]]]
# }
print(object_desc)

config = TransportationEnvConfig(
    world_config= TransportationWorldConfig(
        obstacle_l = [],
        object_l=[
            object_desc
            ],
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
exp_name = 'pos_tol_2_angle_pi18_corridor_10_20_reward_scale_10_success_1_death_1_nn_3x256_triangle'
model = SAC.load('model_ckp/' + exp_name)
print(model.policy)

scene_buffer = CvDrawBuffer(window_name="Simulation", resolution=(1024,1024))

# Manually reset the env to control the initial position of the object
# World reset:
env.world.agent_collided = 0
env.world.object_collided = 0

env.world.goal['pos'].x = 55
env.world.goal['pos'].y = 25
env.world.goal['angle'] = 0

env.world.obj = env.world.obj_l[0]
env.world.obj.obj_rigid_body.position = (15, 25)
env.world.obj.obj_rigid_body.angle = np.pi

# env.world.agent.agent_rigid_body.position = (10, 25)
x_lim = [
    env.world.obj.obj_rigid_body.position.x - env.world.max_obj_dist*0.7071,
    env.world.obj.obj_rigid_body.position.x + env.world.max_obj_dist*0.7071
]
y_lim = [
    env.world.obj.obj_rigid_body.position.y - env.world.max_obj_dist*0.7071,
    env.world.obj.obj_rigid_body.position.y + env.world.max_obj_dist*0.7071
]
env.world.agent.agent_rigid_body.position = env.world.gen_non_overlapping_position_in_limit(
    env.world.agent.agent_radius*1.2, x_lim, y_lim)

env.step_count = 0
env.corridor_width = 12.0
env._update_corridor_variables()

obs = env._gen_observation()

trajectory = []

render()
acc_reward = 0
success_l = []
while True:
    trajectory.append({
        'pos': b2Vec2(env.world.obj.obj_rigid_body.position),
        'angle': env.world.obj.obj_rigid_body.angle
    })
    print(env.world.obj.obj_rigid_body.position, env.world.obj.obj_rigid_body.angle)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    acc_reward += reward
    render()
    if terminated or truncated:
        success_l.append(info['is_success'])
        print()
        print('Reward: ', acc_reward)
        acc_reward = 0
        print('Success rate: ', sum(success_l) / len(success_l))
        break

# Draw the trajectory on a separate window as a sequence of objects
# Skip every skip_n frames
# Center it on the capsule
n_frames = 5
if n_frames > len(trajectory): n_frames = len(trajectory)
n_frames -= 2 # First and last frame
skip_n = int((len(trajectory)-2) / n_frames)

draw_buffer = CvDrawBuffer(window_name="Trajectory", resolution=(1024,1024))
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
cv2.polylines(screen, [points], isClosed=True, color=(0, 0, 0), thickness=3)

# Draw the first
env.world.obj.DrawInPose(
    trajectory[0]['pos'], trajectory[0]['angle'], env.world.pixels_per_meter, screen, (0, 255, 0), -1)
# Draw the last
env.world.obj.DrawInPose(
    env.world.goal['pos'], env.world.goal['angle'], env.world.pixels_per_meter, screen, (0, 0, 255), -1)

# Draw the trajectory as a curve
for i in range(len(trajectory)):
    if i == 0 or i == len(trajectory)-1: continue
    cv2.line(
        screen, 
        env.world.worldToScreen(trajectory[i-1]['pos']), 
        env.world.worldToScreen(trajectory[i]['pos']), 
        color=(0, 0, 0), thickness=2)
cv2.line(
    screen, 
    env.world.worldToScreen(trajectory[-2]['pos']), 
    env.world.worldToScreen(trajectory[-1]['pos']), 
    color=(0, 0, 0), thickness=2)
cv2.line(
    screen, 
    env.world.worldToScreen(trajectory[-1]['pos']), 
    env.world.worldToScreen(env.world.goal['pos']), 
    color=(0, 0, 0), thickness=2)
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


# Draw the trajectory as a sequence of frames
for i in list(range(skip_n, len(trajectory)-1, skip_n)):
    print('i = ', i)
    obj_screen = copy.deepcopy(screen)
    env.world.obj.DrawInPose(
            trajectory[i]['pos'], trajectory[i]['angle'], env.world.pixels_per_meter, obj_screen, (128, 128, 128), -1)
    # DrawInPose(env.world.obj,
    #         trajectory[i]['pos'], trajectory[i]['angle'], env.world.pixels_per_meter, obj_screen, (128, 128, 128), 2)
    
    screen = cv2.addWeighted(obj_screen, 0.5, screen, 0.5, 0)
    
    draw_buffer.PushFrame(screen)
    draw_buffer.Draw()
    cv2.waitKey()

# Center the image on the capsule
s = 2
w = 12
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
cv2.imwrite('img.png', screen)