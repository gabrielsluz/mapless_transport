import gymnasium as gym
from gymnasium import spaces
import numpy as np

from research_envs.b2PushWorld.NavigationWorld import NavigationWorld, NavigationWorldConfig

import dataclasses

@dataclasses.dataclass
class NavigationEnvConfig:
    world_config: NavigationWorldConfig = NavigationWorldConfig()
    max_steps: int = 1000

class NavigationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config: NavigationEnvConfig = NavigationEnvConfig()):
        self.config = config
        self.world = NavigationWorld(config.world_config)
        self.action_space = spaces.Discrete(8)
        # Observation: Laser + agent to final goal vector
        n_rays = config.world_config.n_rays
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_rays+2,), dtype=np.float32)

        self.max_steps = config.max_steps
        self.step_count = 0

    def _gen_observation(self):
        range_l, _, _ = self.world.get_laser_readings()
        laser_readings = np.array(range_l) / self.world.range_max
        
        agent_to_goal = self.world.agent_to_goal_vector()
        agent_to_goal = np.array(agent_to_goal) / agent_to_goal.length
        return np.concatenate((laser_readings, agent_to_goal))

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

    
