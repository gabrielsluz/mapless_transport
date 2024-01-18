import numpy as np
from Box2D import b2PolygonShape, b2Vec2, b2FixtureDef, b2Transform
import math
from math import sin, cos, pi

import cv2

# Agent that take actions according to a potential field
class AgentDirectionPose:
    def __init__(self, simulator = None, b2_shape = None, x = 0, y = 0, velocity = 1.0, maxForceLength = 5.0, minForceLength = 1.0):
        # define the current state of the agent
        # 0 - idle (can receive actions)
        # 1 - performing an action (can do nothing besides wait)
        self.state = 0
        self.current_obj = b2Vec2(0,0)

        self.max_force_length = maxForceLength
        self.min_force_length = minForceLength
        self.velocity = velocity

        # set simulator reference to allow computing global metrics
        self.simulator = simulator

        # ----------- Body creation -----------
        # You can create all bodies to configure
        # them later
        # self.box_def         = b2BodyDef(position = (1,1))
        # self.box_rigid_body  = self.world.CreateBody(self.box_def)
        self.agent_rigid_body  = simulator.world.CreateDynamicBody(position=(x,y), fixedRotation=False)
        self.agent_rigid_body.userData = {'type': 'agent'} # For collision detection

        # ----------- Body configuration ------------
        self.agent_shape = b2PolygonShape(box=(10.0/2, 5.0/2))
        self.agent_radius = math.sqrt((10.0/2)**2 + (5.0/2)**2)
        self.agent_fixture_def = b2FixtureDef(shape=self.agent_shape, density=1, friction=0.2)
        self.agent_rigid_body.CreateFixture(self.agent_fixture_def)
        self.last_agent_pos = b2Vec2(x,y)

        # --- Limit timesteps
        self.cur_action_steps =0 

    def UpdateLastPos(self):
        self.last_agent_pos = self.agent_rigid_body.position.copy()

    def Dist(self, other):
        return (self.agent_rigid_body.position - other).length

    def GetPositionAsList(self):
        return (self.agent_rigid_body.position[0], self.agent_rigid_body.position[1])

    def PerformAction(self, action):
        # spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        #   0 - linear direction => angle
        #   1 - linear velocity
        #   2 - angular velocity

        #Linear
        theta = 2*pi * action[0] 
        direction = np.array([cos(theta), sin(theta)])
        force = self.min_force_length + (self.max_force_length - self.min_force_length) * action[1]
        
        self.current_obj = self.agent_rigid_body.position + (force * direction)
        self.current_obj = b2Vec2(self.current_obj)

        # Angular
        time_to_pos = force / self.velocity
        cur_angle = math.fmod(self.agent_rigid_body.angle, 2*pi)
        if cur_angle < 0.0: cur_angle += 2*pi

        goal_angle = 2*pi * action[2]
        angle_diff = goal_angle - cur_angle
        if angle_diff > np.pi: angle_diff -= 2*np.pi
        elif angle_diff < -np.pi: angle_diff += 2*np.pi

        self.agent_rigid_body.angularVelocity = angle_diff / time_to_pos

        self.state = 1
        self.cur_action_steps = 0

    def IsPerformingAction(self):
        if self.state == 1:
            return True
        return False

    def Update(self):
        if self.state == 0:
            # just wait
            self.agent_rigid_body.linearVelocity = (0,0)
            self.agent_rigid_body.angularVelocity = 0.0
        elif self.state == 1:
            # perform action
            dir = self.current_obj - self.agent_rigid_body.position
            dir.Normalize()
            self.agent_rigid_body.linearVelocity = dir * self.velocity

            # compute error
            err = self.current_obj - self.agent_rigid_body.position
            err = err.length

            if(err < 0.1):
                self.state = 0

            self.cur_action_steps += 1
            if(self.cur_action_steps > 100):
                self.state = 0


    # Draw functions
    def worldToScreen(self, position, pixels_per_meter):
        return (int(position[0] * pixels_per_meter), int(position[1] * pixels_per_meter))
    
    def Draw(self, pixels_per_meter, image, color, thickness):
        body = self.agent_rigid_body
        for f_i in range(len(body.fixtures)):
            vertices = [(body.transform * v) for v in body.fixtures[f_i].shape.vertices]
            vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
            cv2.fillPoly(image, [np.array(vertices)], color)
        
        # Aux:
        v = body.worldCenter
        v = self.worldToScreen(v, pixels_per_meter)
        cv2.circle(image, v, 10, (1, 0, 0, 0), 5)

    def DrawInPos(self, screen_pos, pixels_per_meter, image, color, thickness):
        body = self.agent_rigid_body
        for f_i in range(len(body.fixtures)):
            # Only rotate
            vertices = [(body.transform.q * v) for v in body.fixtures[f_i].shape.vertices]
            vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
            # Translate
            vertices = [(v[0]+screen_pos[0], v[1]+screen_pos[1]) for v in vertices]
            cv2.fillPoly(image, [np.array(vertices)], color)

    def DrawInPose(self, world_pos, angle, pixels_per_meter, image, color, thickness):
        transform_matrix = b2Transform()
        transform_matrix.SetIdentity()
        transform_matrix.Set(world_pos, angle)
        body = self.agent_rigid_body
        for f_i in range(len(body.fixtures)):
            vertices = [(transform_matrix * v) for v in body.fixtures[f_i].shape.vertices]
            vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
            cv2.fillPoly(image, [np.array(vertices)], color)