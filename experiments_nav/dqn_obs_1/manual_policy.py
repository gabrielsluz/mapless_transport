# Include path two levels up
import sys
sys.path.append('../..')

from research_envs.envs.navigation_env import NavigationEnvConfig, NavigationMixEnv
from research_envs.envs.obstacle_repo import obstacle_l_dict
from research_envs.b2PushWorld.NavigationWorld import NavigationWorldConfig
from research_envs.cv_buffer.CvDrawBuffer import CvDrawBuffer

import cv2
import numpy as np

action_to_nl = {
    0: 'right      ',
    1: 'lower right',
    2: 'down       ',
    3: 'lower left ',
    4: 'left       ',
    5: 'upper left ',
    6: 'up         ',
    7: 'upper right'
}

def calc_angle_dist(a0, a1):
    # a0, a1 in [-1, 1]
    # return in [0, 1]
    # Return the smallest distance counter or clockwise
    a0 = a0 if a0 > 0 else 2 + a0
    a1 = a1 if a1 > 0 else 2 + a1

    if a0 > a1:
        bigger = a0
        smaller = a1
    else:
        bigger = a1
        smaller = a0
    
    dist = bigger - smaller
    if dist > 1:
        dist = 2 - dist
    return dist

def min_action_obs_dist(action, obs):
    # Find the 3 closest observations to action:
    # Transforma o id da action em um id da observacao
    action = (action + 4) % 8
    obs_id_l = [(action + i) % 8 for i in [-1, 0, 1]]
    return min([obs[i] for i in obs_id_l])
    
def pick_action(obs):
    """
    Algoritmo manual: checa quais ações mais próximas do objetivo e escolhe a que não tem obstáculo.
    -0.57666373  0.71601754
    obs[8] => angle to goal, mapear para um int de 0 a 7
    obs[0:8] => distâncias para os obstáculos, mapear para um int de 0 a 7

    Observacoes: 0 a -0.25
    """
    action_angles = [1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]
    angle_to_goal = obs[8]
    action_goal_dist = [calc_angle_dist(angle_to_goal, a) for a in action_angles]
    sorted_actions = np.argsort(action_goal_dist)[::-1]
    # Retorna a primeira ação que não tem obstáculo 
    # => se alguma das 3 observacoes estiverem bloqueadas, entao a acao está bloqueada
    # Bloqeada = < 0.06
    action_block_arr = np.zeros(8)
    for action in sorted_actions:
        action_block_arr[action] = min_action_obs_dist(action, obs)
        if min_action_obs_dist(action, obs) > 0.065:
            return action
    return np.argsort(action_block_arr)[0]

def render():
    scene_buffer.PushFrame(env.render())
    scene_buffer.Draw()
    cv2.waitKey(1)

config = NavigationEnvConfig(
    world_config= NavigationWorldConfig(
        obstacle_l = [],
        n_rays = 8,
        range_max = 25.0
    ),
    max_steps = 200,
    previous_obs_queue_len = 0
)
obs_l_dict = {
    k: obstacle_l_dict[k] 
    for k in ['25_circles']#['4_circles', 'corridor', 'crooked_corridor', '16_circles', 'sparse_1', 'sparse_2']
}
env = NavigationMixEnv(config, obs_l_dict)



scene_buffer = CvDrawBuffer(window_name="Simulation", resolution=(1024,1024))

render()
obs, info = env.reset()
acc_reward = 0
success_l = []
while True:
    action = pick_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    acc_reward += reward
    render()
    if terminated or truncated:
        success_l.append(info['is_success'])
        obs, info = env.reset()
        print('Reward: ', acc_reward)
        acc_reward = 0
        print('Success rate: ', sum(success_l) / len(success_l))

