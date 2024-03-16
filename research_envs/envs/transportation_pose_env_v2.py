import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random
from collections import deque
from Box2D import b2Vec2
from shapely.geometry import LineString, Point

from research_envs.b2PushWorld.TransportationPoseWorld import TransportationWorld, TransportationWorldConfig

import dataclasses

import cv2

@dataclasses.dataclass
class TransportationEnvConfig:
    world_config: TransportationWorldConfig = TransportationWorldConfig()
    max_steps: int = 1000
    previous_obs_queue_len: int = 0
    reward_scale: float = 1.0
    max_goal_dist: float = 50.0

"""
The agent must push the object to the goal, staying inside the corridor and two circles.
One circle is centered at the start position of the object and the other at the goal position.
Basically, it should have a small distance to a line segment between the two points.
In this env, we use reward to encourage this.
"""

class TransportationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config: TransportationEnvConfig = TransportationEnvConfig()):
        self.config = config
        self.world = TransportationWorld(config.world_config)
        
        self.action_space = spaces.Box(
            low=0, high=1, shape=(2,), dtype=np.float32)
        # Action only queue
        self.prev_action_queue = deque(maxlen=config.previous_obs_queue_len)
        self.prev_act_len = config.previous_obs_queue_len*2

        self.observation_shape = (
            6 + self.prev_act_len,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32)

        self.max_obj_dist = self.world.max_obj_dist
        self.max_goal_dist = config.max_goal_dist# max(self.world.width, self.world.height)
        
        self.reward_scale = config.reward_scale

        self.max_steps = config.max_steps

        self.reset()

    def _gen_current_observation(self):        
        agent_to_goal = self.world.agent_to_goal_vector()
        # Calc angle between agent_to_goal and x-axis
        angle = np.arctan2(agent_to_goal[1], agent_to_goal[0])
        goal_obs = np.array([
            angle/np.pi, 
            agent_to_goal.length / self.max_goal_dist,
            self.world.goal['angle']/(2*math.pi) # Comment For PositionControl
            ])

        # angle, dist, object angle
        obj_angle = math.fmod(self.world.obj.obj_rigid_body.angle, 2*math.pi)
        agent_to_obj = self.world.agent_to_object_vector()
        # Calc angle between agent_to_obj and x-axis
        angle = np.arctan2(agent_to_obj[1], agent_to_obj[0])
        obj_obs = np.array([
            angle/np.pi, 
            agent_to_obj.length / self.max_obj_dist, 
            obj_angle / (2*np.pi)
            ])

        return np.concatenate((goal_obs, obj_obs), dtype=np.float32)
 
    def _gen_observation(self):
        cur_obs = self._gen_current_observation()
        prev_act = np.zeros(self.prev_act_len)
        for i, a in enumerate(reversed(self.prev_action_queue)):
            prev_act[2*i] = a[0]
            prev_act[2*i+1] = a[1]

        obs = np.concatenate([cur_obs, prev_act], dtype=np.float32)
        return obs

    def _check_success(self):
        return self.world.did_object_reach_goal()

    def _check_death(self):
        if self.world.did_agent_collide() or self.world.did_object_collide():
            return True
        if self.world.agent_to_object_vector().length > self.max_obj_dist:
            return True

    def _calc_reward(self):
        # Reward based on the progress of the agent towards the goal	
        # Limits the maximum reward to [-1.0, 2.0] on average
        success_reward = 1.0
        death_penalty = -1.0	
        time_penalty = -0.01	

        # Terminated
        termination_reward = 0.0
        # Success
        if self._check_success():
            termination_reward =  success_reward
        # Death
        if self._check_death():
            termination_reward =  death_penalty
        # Progress 
        # On average the final progress reward is 1.0 when successful
        cur_dist = self.world.object_to_goal_vector().length
        progress_reward = (self.last_obj_to_goal_d - cur_dist)
        progress_reward = progress_reward / (self.max_goal_dist/2)

        orient_reward = abs(self.last_obj_to_goal_angle_d / np.pi) - abs(self.world.distToOrientation()/np.pi)
        return (
            termination_reward +
            0.5 * progress_reward + 
            0.5 * orient_reward +
            time_penalty)

    def _prepare_before_step(self, action):
        self.prev_action_queue.append(action)
        # Progress to goal
        self.last_obj_to_goal_d = self.world.object_to_goal_vector().length
        self.last_obj_to_goal_angle_d = self.world.distToOrientation()


    def step(self, action):
        # (observation, reward, terminated, truncated, info)
        self._prepare_before_step(action)

        self.world.take_action(action)
        observation = self._gen_observation()
        self.step_count += 1
        
        info = {'is_success': False, "TimeLimit.truncated": False}
        reward = self.reward_scale * self._calc_reward()
        terminated = self._check_success() or self._check_death()
        if self._check_success(): 
            info['is_success'] = True
        truncated = False
        if self.step_count > self.max_steps: 
            info["TimeLimit.truncated"] = True
            truncated = True
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset()
        self.step_count = 0

        self.prev_action_queue.clear()

        return self._gen_observation(), {}

    def render(self, mode='human'):
        return self.world.drawToBufferWithLaser()#self.world.drawToBufferObservation()

    def close(self):
        pass

    def seed(self, seed=None):
        pass


"""
Environment for mixing different NavigationEnv.
Useful when we want to have different obstacle setups.
"""
class TransportationMixEnv(gym.Env):
    def __init__(
        self, 
        config: TransportationEnvConfig = TransportationEnvConfig(), 
        obstacle_l_dict: dict = {'empty':[]}
        ):
        self.env_l = []
        for key in obstacle_l_dict.keys():
            config.world_config.obstacle_l = obstacle_l_dict[key]
            self.env_l.append((key, TransportationEnv(config)))

        idx = np.random.randint(len(self.env_l))
        self.cur_env = self.env_l[idx][1]
        self.cur_env_name = self.env_l[idx][0]

        self.action_space = self.cur_env.action_space
        self.observation_space = self.cur_env.observation_space

    def step(self, action):
        return self.cur_env.step(action)
    
    def reset(self, seed=None, options=None):
        idx = np.random.randint(len(self.env_l))
        self.cur_env = self.env_l[idx][1]
        self.cur_env_name = self.env_l[idx][0]

        return self.cur_env.reset(seed=seed, options=options)
    
    def render(self, mode='human'):
        return self.cur_env.render()

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    