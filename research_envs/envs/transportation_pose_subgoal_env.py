import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random
from collections import deque
from Box2D import b2Vec2, b2Transform
import cv2

from shapely import LineString, Polygon

from research_envs.b2PushWorld.TransportationPoseWorld import TransportationWorld, TransportationWorldConfig

import dataclasses

@dataclasses.dataclass
class TransportationEnvConfig:
    world_config: TransportationWorldConfig = TransportationWorldConfig()
    max_steps: int = 1000
    previous_obs_queue_len: int = 0
    reward_scale: float = 1.0

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


        self.observation_shape = (
            n_rays + 8 + self.prev_act_len,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32)

        self.max_obj_dist = self.world.max_obj_dist
        self.max_goal_dist = max(self.world.width, self.world.height)

        self.reward_scale = config.reward_scale

        # Subgoals parameters
        self.max_subgoal_pos_dist = 20.0
        self.min_subgoal_pos_dist = 15.0
        self.max_subgoal_angle_dist = np.pi/12

        self.obj_vertices = self.world.obj.obj_rigid_body.fixtures[0].shape.vertices

        self.max_steps = config.max_steps
        self.step_count = 0

        self.reset()

    def _gen_subgoal_candidate(self):
        # Gen subgoals in a range radius from the objects position
        rand_dist = random.uniform(self.min_subgoal_pos_dist, self.max_subgoal_pos_dist)
        neg = 1 if random.randint(0, 1) == 0 else -1
        rand_dist = neg * rand_dist
        rand_rad = random.uniform(0, 2*np.pi)
        subgoal_pos = [
            self.world.obj.obj_rigid_body.position.x + rand_dist * np.cos(rand_rad),
            self.world.obj.obj_rigid_body.position.y + rand_dist * np.sin(rand_rad)
        ]

        obj_angle = math.fmod(self.world.obj.obj_rigid_body.angle, 2*math.pi)
        if obj_angle < 0.0: obj_angle += 2*math.pi
        subgoal_angle = obj_angle + random.uniform(-self.max_subgoal_angle_dist, self.max_subgoal_angle_dist)

        return subgoal_pos, subgoal_angle

    def _gen_subgoal(self):
        # Find the forbidden lines => likely to have obstacles
        forbidden_lines = []
        range_l, _, point_l = self.world.get_laser_readings()
        agent_pos = self.world.agent.agent_rigid_body.position
        for i, p in enumerate(point_l):
            if range_l[i] < self.world.range_max:
                # Start point = p
                # End point: line from agent to p extended to range_max
                dir_vec = (p - agent_pos)
                dir_vec.Normalize()
                end_p = agent_pos + dir_vec * self.world.range_max
                forbidden_lines.append((
                    (p[0], p[1]), 
                    (end_p[0], end_p[1])
                ))
        
        sg_candidates = [self._gen_subgoal_candidate() for _ in range(100)]

        # Check if valid:
        min_dist = None
        min_sg = sg_candidates[0]

        transform_matrix = b2Transform()
        transform_matrix.SetIdentity()
        self.is_valid_sg = []
        for sg in sg_candidates:
            # Create polygon on pos and angle, based on the object
            transform_matrix.Set(sg[0], sg[1])
            vertices = [(transform_matrix * v) for v in self.obj_vertices]
            poly = Polygon(vertices)

            # Check if the subgoal is in the forbidden lines
            is_valid = True
            for line in forbidden_lines:
                if poly.intersects(LineString([line[0], line[1]])):
                    is_valid = False
                    break
            self.is_valid_sg.append(is_valid)

            if is_valid:
                dist = (self.world.goal['pos'] - b2Vec2(sg[0])).length
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    min_sg = sg


        self.forbidden_lines = forbidden_lines
        self.sg_candidates = sg_candidates

        return min_sg

    def _gen_current_observation(self):
        range_l, _, _ = self.world.get_laser_readings()
        laser_readings = np.array(range_l) / (self.world.range_max)

        # angle, dist, goal_angle
        agent_to_goal = self.world.agent_to_goal_vector()
        # Calc angle between agent_to_goal and x-axis
        angle = np.arctan2(agent_to_goal[1], agent_to_goal[0])
        goal_obs = np.array([
            angle/np.pi, min(agent_to_goal.length/self.max_goal_dist, 1.0),
            self.world.goal['angle']/(2*math.pi)
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
        
        # Subgoal: angle, dist
        agent_to_subgoal = (b2Vec2(self.subgoal[0]) - self.world.agent.agent_rigid_body.position)
        angle = np.arctan2(agent_to_subgoal[1], agent_to_subgoal[0])
        subgoal_obs = np.array([
            angle/np.pi, min(agent_to_subgoal.length/self.max_goal_dist, 1.0)
            ])

        return np.concatenate((laser_readings, goal_obs, obj_obs, subgoal_obs), dtype=np.float32)
    
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
    
    def _check_death(self):
        obj_dist = self.world.agent_to_object_vector().length
        return self.world.did_agent_collide() or self.world.did_object_collide() or obj_dist > self.max_obj_dist

    def _calc_reward(self):
        # Reward based on the progress of the agent towards the subgoal	
        # Limits the maximum reward to [-1.0, 2.0] on average
        progress_reward = 0.0
        success_reward = 0.5
        death_penalty = -1.0	
        time_penalty = -0.01	

        # Success
        if self._check_success():
            return success_reward
        # Death
        if self._check_death():
            return death_penalty
        # Progress 
        # On average the final progress reward is 1.0 when successful
        cur_dist = self.object_to_subgoal_dist()
        progress_reward = (self.last_dist - cur_dist)
        progress_reward = progress_reward / (self.max_goal_dist/2)

        orient_reward = abs(self.last_orient_error) - abs(self.world.distToOrientation()/np.pi)

        return 0.5*(progress_reward + orient_reward) + time_penalty

    def object_to_subgoal_dist(self):
        return (b2Vec2(self.subgoal[0]) - self.world.obj.obj_rigid_body.position).length

    def step(self, action):
        # (observation, reward, terminated, truncated, info)
        self.prev_action_queue.append(action)
        self.last_dist = self.object_to_subgoal_dist()
        self.last_orient_error = self.world.distToOrientation()/ np.pi

        self.world.take_action(action)
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

        self.subgoal = self._gen_subgoal()
        observation = self._gen_observation()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset()
        self.step_count = 0

        self.prev_action_queue.clear()
        # self.prev_obs_queue.clear()
        self.subgoal = self._gen_subgoal()
        return self._gen_observation(), {}

    def render(self, mode='human'):
        # return self.world.drawToBufferWithLaser()#self.world.drawToBufferObservation()
        screen = self.world.drawToBufferWithLaser()
        subgoal = self.subgoal
        self.world.obj.DrawInPose(subgoal[0], subgoal[1],
                                   self.world.pixels_per_meter, screen, (255, 255, 0), -1)


        for i in range(len(self.sg_candidates)):
            sg = self.sg_candidates[i]
            sg = self.world.worldToScreen(sg[0])
            if self.is_valid_sg[i]:
                cv2.circle(screen, sg, 5, (0, 255, 0), -1)
            else:
                cv2.circle(screen, sg, 5, (255, 0, 0), -1)
        # for sg in self.sg_candidates:
        #     # Draw using cv2.circle
        #     sg = self.world.worldToScreen(sg[0])
        #     cv2.circle(screen, sg, 5, (0, 255, 0), -1)
        # Draw the forbidden_lines
        for line in self.forbidden_lines:
            p1 = self.world.worldToScreen(line[0])
            p2 = self.world.worldToScreen(line[1])
            cv2.line(screen, p1, p2, (255, 0, 0), 1)
        return screen

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

    
