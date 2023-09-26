import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

from research_envs.b2PushWorld.NavigationWorld import NavigationWorld, NavigationWorldConfig

import dataclasses

@dataclasses.dataclass
class NavigationEnvConfig:
    world_config: NavigationWorldConfig = NavigationWorldConfig()
    max_steps: int = 1000
    previous_obs_queue_len: int = 0

class NavigationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config: NavigationEnvConfig = NavigationEnvConfig()):
        self.config = config
        self.world = NavigationWorld(config.world_config)
        self.action_space = spaces.Discrete(8)

        # Observation: Laser + agent to final goal vector
        n_rays = config.world_config.n_rays
        # Observation Queue
        self.prev_obs_queue = deque(maxlen=config.previous_obs_queue_len)
        self.prev_action_queue = deque(maxlen=config.previous_obs_queue_len)
        self.prev_obs_len = config.previous_obs_queue_len * (n_rays+2)
        self.prev_act_len = config.previous_obs_queue_len
        self.observation_shape = (
            n_rays+2 + self.prev_obs_len + self.prev_act_len,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32)

        self.max_steps = config.max_steps
        self.step_count = 0


    def _gen_current_observation(self):
        range_l, _, _ = self.world.get_laser_readings()
        laser_readings = np.array(range_l) / self.world.range_max
        
        agent_to_goal = self.world.agent_to_goal_vector()
        # Calc angle between agent_to_goal and x-axis
        angle = np.arctan2(agent_to_goal[1], agent_to_goal[0])
        goal_obs = np.array([
            angle/np.pi, agent_to_goal.length/25.0
            ])
        return np.concatenate((laser_readings, goal_obs))
    
    def _gen_observation(self):
        cur_obs = self._gen_current_observation()

        prev_obs = np.zeros(self.prev_obs_len)
        aux = []
        for obs in reversed(self.prev_obs_queue):
            aux += list(obs)
        prev_obs[:len(aux)] = aux

        prev_act = np.zeros(self.prev_act_len)
        aux = [
            (a+1)/(self.action_space.n+1) # Avoid a = 0
            for a in reversed(self.prev_action_queue)
        ]
        prev_act[:len(aux)] = aux

        obs = np.concatenate([cur_obs, prev_obs, prev_act])
        self.prev_obs_queue.append(cur_obs)
        return obs

    def _calc_reward(self):
        if self.world.did_agent_collide():
            return -1
        elif self.world.did_agent_reach_goal():
            return 2
        else:
            return -0.01

    def step(self, action):
        # (observation, reward, terminated, truncated, info)
        self.world.take_action(action)
        self.prev_action_queue.append(action)
        observation = self._gen_observation()
        self.step_count += 1
        
        info = {'is_success': False}
        reward = self._calc_reward()
        terminated = self.world.did_agent_collide() or self.world.did_agent_reach_goal()
        if self.world.did_agent_reach_goal(): 
            info['is_success'] = True
        truncated = self.step_count > self.max_steps
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset()
        self.step_count = 0

        self.prev_action_queue.clear()
        self.prev_obs_queue.clear()
        return self._gen_observation(), {}

    def render(self, mode='human'):
        return self.world.drawToBufferWithLaser()

    def close(self):
        pass

    def seed(self, seed=None):
        pass


"""
Environment for mixing different NavigationEnv.
Useful when we want to have different obstacle setups.
"""
class NavigationMixEnv(gym.Env):
    def __init__(
        self, 
        config: NavigationEnvConfig = NavigationEnvConfig(), 
        obstacle_l_dict: dict = {'empty':[]}
        ):
        self.env_l = []
        for key in obstacle_l_dict.keys():
            config.world_config.obstacle_l = obstacle_l_dict[key]
            self.env_l.append(NavigationEnv(config))
        self.action_space = self.env_l[0].action_space
        self.observation_space = self.env_l[0].observation_space
        self.cur_env = self.env_l[np.random.randint(len(self.env_l))]

    def step(self, action):
        return self.cur_env.step(action)
    
    def reset(self, seed=None, options=None):
        self.cur_env = self.env_l[np.random.randint(len(self.env_l))]
        return self.cur_env.reset(seed=seed, options=options)
    
    def render(self, mode='human'):
        return self.cur_env.render()

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    
