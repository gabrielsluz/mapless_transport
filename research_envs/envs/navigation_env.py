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
            low=0.0, high=1.0, shape=(n_rays+2, ), dtype=np.float32)

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
            return 1
        else:
            return 0

    def step(self, action):
        # (observation, reward, terminated, truncated, info)
        self.world.take_action(action)
        observation = self._gen_observation()
        
        reward = self._calc_reward()
        terminated = self.world.did_agent_collide() or self.world.did_agent_reach_goal()
        truncated = self.step_count >= self.max_steps
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.world.reset()
        self.step_count = 0
        return self._gen_observation()

    def render(self, mode='human'):
        return self.world.drawToBufferWithLaser()

    def close(self):
        pass

    def seed(self, seed=None):
        pass