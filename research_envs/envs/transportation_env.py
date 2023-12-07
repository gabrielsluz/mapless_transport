import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from collections import deque
from Box2D import b2Vec2

from research_envs.b2PushWorld.TransportationWorld import TransportationWorld, TransportationWorldConfig

import dataclasses

@dataclasses.dataclass
class TransportationEnvConfig:
    world_config: TransportationWorldConfig = TransportationWorldConfig()
    max_steps: int = 1000
    previous_obs_queue_len: int = 0

class TransportationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config: TransportationEnvConfig = TransportationEnvConfig()):
        self.config = config
        self.world = TransportationWorld(config.world_config)
        if config.world_config.agent_type == 'discrete':
            self.action_space = spaces.Discrete(self.world.agent.directions)
        elif config.world_config.agent_type == 'continuous':
            self.action_space = spaces.Box(
                low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observation: Laser + agent to final goal vector
        n_rays = config.world_config.n_rays
        # Observation Queue
        # self.prev_obs_queue = deque(maxlen=config.previous_obs_queue_len)
        # self.prev_action_queue = deque(maxlen=config.previous_obs_queue_len)
        # self.prev_obs_len = config.previous_obs_queue_len * (n_rays+2)
        # self.prev_act_len = config.previous_obs_queue_len
        # self.observation_shape = (
        #     n_rays+2 + self.prev_obs_len + self.prev_act_len,
        # )
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32)
        
        # Action only queue
        self.prev_action_queue = deque(maxlen=config.previous_obs_queue_len)

        if config.world_config.agent_type == 'discrete':
            self.prev_act_len = config.previous_obs_queue_len
        elif config.world_config.agent_type == 'continuous':
            self.prev_act_len = config.previous_obs_queue_len*2


        self.observation_shape = (
            n_rays + 5 + self.prev_act_len,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32)

        self.max_obj_dist = self.world.max_obj_dist
        self.max_goal_dist = max(self.world.width, self.world.height)

        self.max_steps = config.max_steps
        self.step_count = 0


    def _gen_current_observation(self):
        range_l, _, _ = self.world.get_laser_readings()
        laser_readings = np.array(range_l) / (self.world.range_max)

        # laser_readings = []
        # _, _, point_l = self.world.get_laser_readings()
        # agent_pos = self.world.agent.agent_rigid_body.position
        # for p in point_l:
        #     agent_to_p = b2Vec2(p) - agent_pos
        #     # laser_readings.append(np.arctan2(agent_to_p[1], agent_to_p[0])/np.pi)
        #     laser_readings.append(agent_to_p.length / self.world.range_max)

        # angle, dist
        agent_to_goal = self.world.agent_to_goal_vector()
        # Calc angle between agent_to_goal and x-axis
        angle = np.arctan2(agent_to_goal[1], agent_to_goal[0])
        goal_obs = np.array([
            angle/np.pi, min(agent_to_goal.length/self.max_goal_dist, 1.0)
            ])
        
        # angle, dist, object angle
        obj_angle = math.fmod(self.world.obj.obj_rigid_body.angle, 2*math.pi)
        if obj_angle < 0.0: obj_angle += 2*math.pi
        obj_angle = obj_angle / (2*math.pi)
        
        agent_to_obj = self.world.agent_to_object_vector()
        # Calc angle between agent_to_obj and x-axis
        angle = np.arctan2(agent_to_obj[1], agent_to_obj[0])
        obj_obs = np.array([
            angle/np.pi, min(agent_to_obj.length/self.max_obj_dist, 1.0), obj_angle
            ])

        return np.concatenate((laser_readings, goal_obs, obj_obs), dtype=np.float32)
    
    def _gen_observation(self):
        cur_obs = self._gen_current_observation()

        # prev_obs = np.zeros(self.prev_obs_len)
        # aux = []
        # for obs in reversed(self.prev_obs_queue):
        #     aux += list(obs)
        # prev_obs[:len(aux)] = aux

        prev_act = np.zeros(self.prev_act_len)
        aux = [
            (a+1)/(self.action_space.n+1) # Avoid a = 0
            for a in reversed(self.prev_action_queue)
        ]
        prev_act[:len(aux)] = aux

        obs = np.concatenate([cur_obs, prev_act], dtype=np.float32)
        # obs = np.concatenate([cur_obs, prev_obs, prev_act])
        # self.prev_obs_queue.append(cur_obs)
        return obs

    def _check_success(self):
        return self.world.did_object_reach_goal()
    
    def _check_death(self):
        obj_dist = self.world.agent_to_object_vector().length
        return self.world.did_agent_collide() or self.world.did_object_collide() or obj_dist > self.max_obj_dist

    def _calc_reward(self):
        # Reward based on the progress of the agent towards the goal	
        # Limits the maximum reward to [-1.0, 1.0] (except for success or death)
        progress_reward = 0.0
        success_reward = 100.0	
        death_penalty = -100.0	
        time_penalty = -0.01	

        # Success
        if self._check_success():
            return success_reward
        # Death
        if self._check_death():
            return death_penalty
        # Progress
        cur_dist = self.world.object_to_goal_vector().length
        # Tries to scale between -1 and +1, but also clips it	
        max_gain = 2.0 # Heuristic, should be adapted to the environment	
        progress_reward = (self.last_dist - cur_dist) / max_gain  	
        progress_reward = max(min(progress_reward, 1.0), -1.0)	
        return progress_reward + time_penalty

    # def _calc_reward(self):
    #     # Success
    #     if self._check_success():
    #         return 100.0
    #     # Death
    #     if self._check_death():
    #         return -100
    #     return -0.01

    def step(self, action):
        # (observation, reward, terminated, truncated, info)
        self.prev_action_queue.append(action)
        self.last_dist = self.world.object_to_goal_vector().length

        self.world.take_action(action)
        observation = self._gen_observation()
        self.step_count += 1
        
        info = {'is_success': False, "TimeLimit.truncated": False}
        reward = self._calc_reward()
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
        # self.prev_obs_queue.clear()
        self.last_dist = self.world.object_to_goal_vector().length
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

    
