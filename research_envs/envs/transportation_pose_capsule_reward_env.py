import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
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
    reference_corridor_width: float = 10.0

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
        if config.world_config.agent_type == 'discrete':
            self.action_space = spaces.Discrete(self.world.agent.directions)
        elif config.world_config.agent_type == 'continuous':
            self.action_space = spaces.Box(
                low=0, high=1, shape=(2,), dtype=np.float32)

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
        self.agent_type = config.world_config.agent_type


        # self.observation_shape = (
        #     n_rays + 6 + self.prev_act_len,
        # )
        self.observation_shape = (
            7,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32)

        # self.max_obj_dist = self.world.max_obj_dist
        self.max_goal_dist = config.max_goal_dist# max(self.world.width, self.world.height)
        self.reference_corridor_width = config.reference_corridor_width

        self.reward_scale = config.reward_scale

        self._update_corridor_variables()

        self.max_steps = config.max_steps
        self.step_count = 0

    
    def _update_corridor_variables(self):
        # Corridor variables
        self.start_obj_pos = b2Vec2(self.world.obj.obj_rigid_body.position)

        self.capsule_line = LineString([
            (self.start_obj_pos.x, self.start_obj_pos.y),
            (self.world.goal['pos'].x, self.world.goal['pos'].y)
        ])


    # def _gen_current_observation(self):
    #     range_l, _, _ = self.world.get_laser_readings()
    #     laser_readings = np.array(range_l) / (self.world.range_max)

    #     # angle, dist, goal_angle
    #     agent_to_goal = self.world.agent_to_goal_vector()
    #     # Calc angle between agent_to_goal and x-axis
    #     angle = np.arctan2(agent_to_goal[1], agent_to_goal[0])
    #     goal_obs = np.array([
    #         angle/np.pi, min(agent_to_goal.length/self.max_goal_dist, 1.0),
    #         self.world.goal['angle']/(2*math.pi)
    #         ])
        
    #     # angle, dist, object angle
    #     obj_angle = math.fmod(self.world.obj.obj_rigid_body.angle, 2*math.pi)
    #     if obj_angle < 0.0: obj_angle += 2*math.pi
    #     obj_angle = obj_angle / (2*math.pi)
        
    #     agent_to_obj = self.world.agent_to_object_vector()
    #     # Calc angle between agent_to_obj and x-axis
    #     angle = np.arctan2(agent_to_obj[1], agent_to_obj[0])
    #     obj_obs = np.array([
    #         angle/np.pi, 
    #         agent_to_obj.length / self.reference_corridor_width, 
    #         obj_angle
    #         ])

    #     return np.concatenate((laser_readings, goal_obs, obj_obs), dtype=np.float32)

    def _gen_current_observation(self):
        # Only sees the start_obj_pos and goal
        v = self.start_obj_pos - self.world.agent.agent_rigid_body.position
        angle = np.arctan2(v[1], v[0])
        start_obs = np.array([
            angle/np.pi, 
            v.length / self.reference_corridor_width])
        
        agent_to_goal = self.world.agent_to_goal_vector()
        # Calc angle between agent_to_goal and x-axis
        angle = np.arctan2(agent_to_goal[1], agent_to_goal[0])
        goal_obs = np.array([
            angle/np.pi, min(agent_to_goal.length/self.max_goal_dist, 1.0)
            ])

        # angle, dist, object angle
        obj_angle = math.fmod(self.world.obj.obj_rigid_body.angle, 2*math.pi)
        agent_to_obj = self.world.agent_to_object_vector()
        # Calc angle between agent_to_obj and x-axis
        angle = np.arctan2(agent_to_obj[1], agent_to_obj[0])
        obj_obs = np.array([
            angle/np.pi, 
            agent_to_obj.length / self.reference_corridor_width, 
            obj_angle / (2*np.pi)
            ])

        return np.concatenate((start_obs, goal_obs, obj_obs), dtype=np.float32)

    
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
            prev_act = np.zeros(self.prev_act_len)
            for i, a in enumerate(reversed(self.prev_action_queue)):
                prev_act[2*i] = a[0]
                prev_act[2*i+1] = a[1]

        obs = np.concatenate([cur_obs, prev_act], dtype=np.float32)
        # obs = np.concatenate([cur_obs, prev_obs, prev_act])
        # self.prev_obs_queue.append(cur_obs)
        return obs

    def _check_success(self):
        return self.world.did_object_reach_goal()

    # def _check_success(self):
    #     return (self.world.goal['pos'] - self.world.agent.agent_rigid_body.position).length < 1.0

    def _check_death(self):
        return self.world.did_agent_collide() or self.world.did_object_collide()
    
    def _agent_to_corridor_dist(self):
        d = self.capsule_line.distance(
            Point(self.world.agent.agent_rigid_body.position.x,
                  self.world.agent.agent_rigid_body.position.y))
        return d + self.world.agent.agent_radius

    def _calc_reward(self):
        time_penalty = -0.01
        # Success
        if self._check_success():
            return 0.5
        # Progress 
        # On average the final progress reward is 1.0 when successful
        cur_dist = self.world.object_to_goal_vector().length
        progress_reward = (self.last_dist - cur_dist)
        progress_reward = progress_reward / (self.max_goal_dist/2)
        # Corridor penalty
        # d = self._agent_to_corridor_dist()
        # if d >= self.reference_corridor_width:
        #     agent_corr_penalty = -0.5
        # else:
        #     agent_corr_penalty = 0.02 * (-d / self.reference_corridor_width)
        agent_corr_penalty = 0.0
        # Stay close to object
        d = self.world.agent_to_object_vector().length
        if d >= self.reference_corridor_width:
            #agent_obj_penalty = -0.5
            agent_obj_penalty = -0.1
        else:
            agent_obj_penalty = 0.0

        return progress_reward + agent_corr_penalty + agent_obj_penalty + time_penalty

    # def _calc_reward(self):
    #     # Reward based on the progress of the agent towards the goal
    #     progress_reward = 0.0
    #     success_reward = 0.5
    #     death_penalty = -1.0
    #     time_penalty = -0.01

    #     # Success
    #     if self._check_success():
    #         return success_reward
    #     # Death
    #     if self._check_death():
    #         return death_penalty
    #     # Progress 
    #     # On average the final progress reward is 1.0 when successful
    #     cur_dist = self.world.object_to_goal_vector().length
    #     progress_reward = (self.last_dist - cur_dist)
    #     progress_reward = progress_reward / (self.max_goal_dist/2)

    #     orient_reward = abs(self.last_orient_error) - abs(self.world.distToOrientation()/np.pi)

    #     # # Agent to object distance: Agent should stay close to the object
    #     # # Attempt to scale it to sum to -1 on average per episode.
    #     # dist_obj = self.world.agent_to_object_vector().length
    #     # if dist_obj >= self.reference_corridor_width:
    #     #     agent_obj_penalty = -1.0
    #     # else:
    #     #     agent_obj_penalty = 0.05 * (-dist_obj / self.reference_corridor_width)

    #     # Agent to corridor reward
    #     d = self._agent_to_corridor_dist()
    #     if d >= self.reference_corridor_width:
    #         agent_corr_penalty = -1.0
    #     else:
    #         agent_corr_penalty = 0.05 * (-d / self.reference_corridor_width)
    #     # # Corridor Penalty: 
    #     # corr_multiplier = 0.3 # Corr penalty in [-corr_multiplier, 0]
    #     # dist_obj_corr = self.distance_point_to_line(
    #     #     self.world.obj.obj_rigid_body.position,
    #     #     self.start_obj_pos,
    #     #     self.world.goal['pos']
    #     #     )
    #     # corr_penalty = -corr_multiplier * min(1.0, dist_obj_corr / self.max_corr_width)

    #     return 0.5*(progress_reward + orient_reward) + agent_corr_penalty + time_penalty

    def _prepare_before_step(self, action):
        self.prev_action_queue.append(action)
        self.last_dist = self.world.object_to_goal_vector().length
        self.last_orient_error = self.world.distToOrientation()/ np.pi

    # def _prepare_before_step(self, action):
    #     self.prev_action_queue.append(action)
    #     self.last_dist = self.world.agent_to_goal_vector().length


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
        # self.prev_obs_queue.clear()

        self._update_corridor_variables()

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

    
