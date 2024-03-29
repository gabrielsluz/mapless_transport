import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from Box2D import b2Vec2

from research_envs.b2PushWorld.NavigationWorld import NavigationWorld, NavigationWorldConfig

import dataclasses

@dataclasses.dataclass
class NavigationEnvConfig:
    world_config: NavigationWorldConfig = NavigationWorldConfig()
    max_steps: int = 1000
    previous_obs_queue_len: int = 0
    reward_scale: float = 1.0

class NavigationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config: NavigationEnvConfig = NavigationEnvConfig()):
        self.config = config
        self.world = NavigationWorld(config.world_config)

        self.agent_type = config.world_config.agent_type
        if config.world_config.agent_type == 'discrete':
            self.action_space = spaces.Discrete(self.world.agent.directions)
        elif config.world_config.agent_type == 'continuous':
            self.action_space = spaces.Box(
                low=0, high=1, shape=(2,), dtype=np.float32)
        else:
            print('Agent type not implemented in env: ', config.world_config.agent_type)
            return

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
        self.prev_act_len = config.previous_obs_queue_len
        if self.agent_type == 'discrete':
            self.observation_shape = (
                n_rays+2 + self.prev_act_len,
            )
        elif self.agent_type == 'continuous':
            self.observation_shape = (
                n_rays + 2 + 2*self.prev_act_len,
            )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32)
        
        self.max_goal_dist = max(self.world.width, self.world.height)

        self.reward_scale = config.reward_scale

        # Repulsion
        # self.repulsion_min_dist = 5.0
        self.repulsion_max_dist = config.world_config.range_max
        self.repulsion_max_motion_len = 1.0
        self.repulsion_max_angle_diff = np.pi / 4
        self.n_rays = n_rays
        # Normalize such that repulsion_f = 1.0 at maximum
        # We consider 180 degrees => maximum repulsion rays = number of rays / 2
        # Dividing into two equal parts of 90 degrees => n_rays / 4
        # Ponderated by cossine similarity of each ray
        ang_step_sz = (self.repulsion_max_angle_diff) / (self.n_rays / ((2*np.pi)/self.repulsion_max_angle_diff))
        self.max_repulsion_f = 2 * np.sum(np.cos(np.arange(0, self.repulsion_max_angle_diff, ang_step_sz)))

        self.max_steps = config.max_steps
        self.step_count = 0


    def _gen_current_observation(self):
        laser_readings = np.array(self.range_l) / (self.world.range_max)
        
        agent_to_goal = self.world.agent_to_goal_vector()
        # Calc angle between agent_to_goal and x-axis
        angle = np.arctan2(agent_to_goal[1], agent_to_goal[0])
        goal_obs = np.array([
            angle/np.pi, min(agent_to_goal.length/self.max_goal_dist, 1.0),
            ])

        return np.concatenate((laser_readings, goal_obs), dtype=np.float32)
    
    def _gen_observation(self):
        cur_obs = self._gen_current_observation()

        # prev_obs = np.zeros(self.prev_obs_len)
        # aux = []
        # for obs in reversed(self.prev_obs_queue):
        #     aux += list(obs)
        # prev_obs[:len(aux)] = aux

        if self.agent_type == 'discrete':
            prev_act = np.zeros(self.prev_act_len)
            aux = [
                (a+1)/(self.action_space.n+1) # Avoid a = 0
                for a in reversed(self.prev_action_queue)
            ]
            prev_act[:len(aux)] = aux
        elif self.agent_type == 'continuous':
            prev_act = np.zeros(2*self.prev_act_len)

            for i, a in enumerate(reversed(self.prev_action_queue)):
                prev_act[2*i] = a[0]
                prev_act[2*i+1] = a[1]

        obs = np.concatenate([cur_obs, prev_act], dtype=np.float32)
        # obs = np.concatenate([cur_obs, prev_obs, prev_act])
        # self.prev_obs_queue.append(cur_obs)
        return obs
    
    def _calc_repulsion(self):
        agent_pos = self.world.agent.agent_rigid_body.position
        # Find motion direction
        motion_dir = agent_pos - self.last_agent_pos
        motion_len = motion_dir.length
        motion_dir.Normalize()

        # Ponderate laser readings
        repulsion_f = 0.0
        for p in self.point_l:
            p_dir = (p - agent_pos)
            p_len = p_dir.length
            p_dir.Normalize()
            cos_sim = np.dot(motion_dir, p_dir)
            # Simple: binary repulsion
            # if cos_sim > 0.0 and p_len < self.repulsion_min_dist:
            #     repulsion_f += cos_sim
            # Ponderated by inverse distance
            if cos_sim >= np.cos(self.repulsion_max_angle_diff):
                # Squared repulsion
                repulsion_f += cos_sim * (1.0  - p_len / self.repulsion_max_dist)**2
                # repulsion_f += cos_sim * (1.0  - p_len / self.repulsion_max_dist)
        repulsion_f /= self.max_repulsion_f

        # ponderate by motion length
        return repulsion_f * (motion_len / self.repulsion_max_motion_len)

    def _calc_reward(self):
        if self.world.did_agent_collide():
            return -1.0
        elif self.world.did_agent_reach_goal():
            return 1.0
        else:
            repulsion_f = self._calc_repulsion()
            # # Progress reward => getting closer to goal
            # self.cur_dist = self.world.agent_to_goal_vector().length
            # progress_reward = (self.last_dist - self.cur_dist)
            # return progress_reward - 0.2 # time penalty
            return -0.01 - (0.05 * repulsion_f)
        
    def _set_laser_readings(self):
        self.range_l, self.type_l, self.point_l = self.world.get_laser_readings()

    def step(self, action):
        # (observation, reward, terminated, truncated, info)
        self.prev_action_queue.append(action)
        self.last_dist = self.world.agent_to_goal_vector().length
        self.last_agent_pos = b2Vec2(self.world.agent.agent_rigid_body.position)

        self.world.take_action(action)
        self._set_laser_readings()
        observation = self._gen_observation()
        self.step_count += 1
        
        info = {'is_success': False, "TimeLimit.truncated": False}
        reward = self.reward_scale * self._calc_reward()
        terminated = self.world.did_agent_collide() or self.world.did_agent_reach_goal()
        if self.world.did_agent_reach_goal(): 
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
        self._set_laser_readings()
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

    
