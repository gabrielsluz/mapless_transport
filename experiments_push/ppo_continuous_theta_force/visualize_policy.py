# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.b2PushWorld.TransportationWorld import TransportationWorldConfig
from research_envs.envs.transportation_env import TransportationEnvConfig, TransportationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

import cv2
from stable_baselines3 import PPO, SAC

def render():
    # pass
    scene_buffer.PushFrame(env.render())
    scene_buffer.Draw()
    cv2.waitKey(1)


config = TransportationEnvConfig(
    world_config= TransportationWorldConfig(
        obstacle_l = [],
        object_l=[
            {'name': 'MultiPolygons', 'poly_vertices_l':[[[0, 0], [0, 4], [12, 4], [12, 0]], [[0, 4], [0, 8], [4, 8], [4,4]]]}
            # {'name': 'Rectangle', 'height': 10.0, 'width': 5.0}
        ],
        n_rays = 0,
        agent_type = 'continuous',
        force_length=1.0
    ),
    max_steps = 500,
    previous_obs_queue_len = 0
)

obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in [
        'empty'
    ]
}
env = TransportationMixEnv(config, obs_l_dict)

# model = PPO.load("model_ckp/L_min_force_length_025")
model = SAC.load("model_ckp/sac_L_min_force_length_025")
print(model.policy)

scene_buffer = CvDrawBuffer(window_name="Simulation", resolution=(1024,1024))

render()
obs, info = env.reset()
acc_reward = 0
success_l = []
while True:
    action, _states = model.predict(obs, deterministic=True)
    # action = env.cur_env.world.agent.GetRandomValidAction()
    obs, reward, terminated, truncated, info = env.step(action)
    acc_reward += reward
    render()
    if terminated or truncated:
        success_l.append(info['is_success'])
        obs, info = env.reset()
        print('Reward: ', acc_reward)
        acc_reward = 0
        print('Success rate: ', sum(success_l) / len(success_l))

