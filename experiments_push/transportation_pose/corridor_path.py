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
        goal_tolerance={'pos':2, 'angle':np.pi/18}
    ),
    max_steps = 100,
    previous_obs_queue_len = 0,
    reward_scale=10.0
)

obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        'empty',
        # 'empty', 'frame', 'horizontal_corridor', 'vertical_corridor', '4_circles_wide',
        # '1_circle', '1_rectangle', '1_triangle',
        # 'circle_line', 'small_4_circles',
        # '4_circles', 'sparse_1', 'sparse_2',
    ]
}
env = TransportationMixEnv(config, obs_l_dict)

model = SAC.load('model_ckp/progress_sac_rectangle_tolerance_pi18_pos_tol_2_reward_scale_10_corridor_death_width_5')
print(model.policy)

scene_buffer = CvDrawBuffer(window_name="Simulation", resolution=(1024,1024))

goal_l = [
    {'pos':b2Vec2(10,10), 'angle': 0.0},
    {'pos':b2Vec2(20,20), 'angle': 0.0},
    {'pos':b2Vec2(20,40), 'angle': 0.0},
    {'pos':b2Vec2(40,20), 'angle': 0.0},
    {'pos':b2Vec2(40,40), 'angle': 0.0}
]
goal_i = 0

render()
env.reset()
obs, info = set_new_goal(env.cur_env, goal_l[goal_i])

acc_reward = 0
success_l = []
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    acc_reward += reward
    render()
    if terminated or truncated:
        success_l.append(info['is_success'])
        goal_i += 1
        obs, info = set_new_goal(env.cur_env, goal_l[goal_i])
        print('Reward: ', acc_reward)
        acc_reward = 0
        print('Success rate: ', sum(success_l) / len(success_l))

