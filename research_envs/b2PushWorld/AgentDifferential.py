import numpy as np
from Box2D import b2PolygonShape, b2Vec2, b2FixtureDef
from math import sin, cos, pi, sqrt
import cv2

class AgentDifferential:
    def __init__(self, simulator = None, x = 0, y = 0, width=2.0, height=3.0, action_len_steps=10, scale_v=1.0, scale_w=1.0):
        # define the current state of the agent
        # 0 - idle (can receive actions)
        # 1 - performing an action (can do nothing besides wait)
        self.state = 0
        self.current_obj = b2Vec2(0,0)

        # set simulator reference to allow computing global metrics
        self.simulator = simulator

        # ----------- Body creation -----------
        # You can create all bodies to configure
        # them later
        # self.box_def         = b2BodyDef(position = (1,1))
        # self.box_rigid_body  = self.world.CreateBody(self.box_def)
        self.agent_rigid_body  = simulator.world.CreateDynamicBody(position=(x,y))
        self.agent_rigid_body.userData = {'type': 'agent'} # For collision detection

        # ----------- Body configuration ------------
        self.agent_radius = sqrt((width/2)**2 + (height/2)**2)
        self.agent_shape = b2PolygonShape(box=(width/2, height/2))
        self.agent_fixture_def = b2FixtureDef(shape=self.agent_shape, density=1, friction=0.5)
        self.agent_rigid_body.CreateFixture(self.agent_fixture_def)
        self.last_agent_pos = b2Vec2(x,y)

        self.scale_v = scale_v
        self.scale_w = scale_w

        # --- Limit timesteps
        self.action_len_steps = action_len_steps
        self.cur_action_steps =0 

    def UpdateLastPos(self):
        self.last_agent_pos = self.agent_rigid_body.position.copy()

    def Dist(self, other):
        return (self.agent_rigid_body.position - other).length

    def GetPositionAsList(self):
        return (self.agent_rigid_body.position[0], self.agent_rigid_body.position[1])

    def PerformAction(self, unicycle_velocity):
        # unicycle_velocity => [v, w]
        agent_dir = b2Vec2(cos(self.agent_rigid_body.angle), sin(self.agent_rigid_body.angle))
        self.agent_rigid_body.linearVelocity = agent_dir * unicycle_velocity[0] * self.scale_v
        self.agent_rigid_body.angularVelocity = unicycle_velocity[1] * self.scale_w
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
            self.cur_action_steps += 1
            if(self.cur_action_steps >= self.action_len_steps):
                self.state = 0

    def worldToScreen(self, position, pixels_per_meter):
        return (int(position[0] * pixels_per_meter), int(position[1] * pixels_per_meter))

    def Draw(self, pixels_per_meter, image, color, thickness):
        vertices = [(self.agent_rigid_body.transform * v) for v in self.agent_rigid_body.fixtures[0].shape.vertices]
        vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
        cv2.fillPoly(image, [np.array(vertices)], color)