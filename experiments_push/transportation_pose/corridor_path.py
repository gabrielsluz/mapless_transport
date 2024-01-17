# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.b2PushWorld.TransportationPoseWorld import TransportationWorldConfig
from research_envs.envs.transportation_pose_corridor_env import TransportationEnvConfig, TransportationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

import cv2
import numpy as np
from stable_baselines3 import SAC

from Box2D import b2Vec2

def render():
    # pass
    scene_buffer.PushFrame(env.render())
    scene_buffer.Draw()
    cv2.waitKey(100)

def set_goal_on_click(event,x,y,flags,param):
    global wait_bool, obs, info
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert from image coordinates to world coordinates
        goal = {'pos':b2Vec2(
            x / env.cur_env.world.pixels_per_meter, 
            y / env.cur_env.world.pixels_per_meter), 'angle': 0.0}
        obs, info = set_new_goal(env.cur_env, goal)
        wait_bool = False
        # cv2.circle(img,(x,y),100,(255,0,0),-1)
        # mouseX,mouseY = x,y

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

object_desc = {'name': 'Rectangle', 'height': 10.0, 'width': 5.0}
print(object_desc)

config = TransportationEnvConfig(
    world_config= TransportationWorldConfig(
        obstacle_l = [],
        object_l=[
            object_desc
            ],
        n_rays = 0,
        range_max = 25.0,
        agent_type = 'continuous',
        max_force_length=5.0,
        min_force_length=0.1,
        goal_tolerance={'pos':2, 'angle':np.pi/18},
        max_obj_dist=14.0
    ),
    max_steps = 100,
    previous_obs_queue_len = 0,
    reward_scale=10.0,
    max_corr_width=15.0
)

obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        '1_triangle'
        # 'empty', 'frame', 'horizontal_corridor', 'vertical_corridor', '4_circles_wide',
        # '1_circle', '1_rectangle', '1_triangle',
        # 'circle_line', 'small_4_circles',
        # '4_circles', 'sparse_1', 'sparse_2',
    ]
}
env = TransportationMixEnv(config, obs_l_dict)

model = SAC.load('model_ckp/progress_sac_rectangle_tolerance_pi18_pos_tol_2_reward_scale_10_corridor_full_death_width_15')
print(model.policy)

scene_buffer = CvDrawBuffer(window_name="Simulation", resolution=(1024,1024))

render()
cv2.setMouseCallback("Simulation",set_goal_on_click)

obs, info = env.reset()

wait_bool = False

acc_reward = 0
success_l = []
while True:
    if wait_bool:
        render()
        continue
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    acc_reward += reward
    render()
    if terminated or truncated:
        success_l.append(info['is_success'])
        # goal_i += 1
        # obs, info = set_new_goal(env.cur_env, goal_l[goal_i])
        print('Reward: ', acc_reward)
        acc_reward = 0
        print('Success rate: ', sum(success_l) / len(success_l))
        wait_bool = True

