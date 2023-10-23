# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.b2PushWorld.NavigationWorld import NavigationWorldConfig, NavigationWorld, LaserHit
from research_envs.envs.obstacle_repo import obstacle_l_dict

from Box2D import b2Vec2
import numpy as np
import matplotlib.pyplot as plt
import cv2

def drawArrow(image, world_pos, force, color):
    start_pos = world.worldToScreen(world_pos)
    end_pos = world_pos + force
    end_pos = world.worldToScreen(end_pos)
    cv2.arrowedLine(image, start_pos, end_pos, color, thickness=2, tipLength=0.5)


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


world_config=NavigationWorldConfig(
    obstacle_l = obstacle_l_dict['1_triangle'],
    n_rays = 24,
    range_max = 25.0
)

world = NavigationWorld(world_config)

# Iterate through a grid, storing forces
force_dict = {}
step_sz = 1
for x_i in range(0, world.height, step_sz):
    for y_i in range(0, world.width, step_sz):
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
        force_dict[(x_i, y_i)] = att_force + rep_force
        # force_dict[(x_i, y_i)] = rep_force

screen = world.drawToBuffer()

for k in force_dict.keys():
    if np.linalg.norm(force_dict[k]) == 0.0: continue
    start_pos = np.array(k)
    direction = force_dict[k] / np.linalg.norm(force_dict[k])
    direction = step_sz/2 * direction
    drawArrow(screen, start_pos, direction, (0.0, 0.0, 1.0))

# range_l, type_l, point_l = world.get_laser_readings_from_point(b2Vec2(15, 15))
# print(range_l, type_l, point_l)

# screen = world.drawToBuffer()
# drawArrow(screen, np.array([0, 0]), np.array([15, 15]), (0.0, 0.0, 1.0))
plt.imshow(screen)
plt.show(block=True)
# while True:
#     pass

