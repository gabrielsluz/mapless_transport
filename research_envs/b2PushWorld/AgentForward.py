import numpy as np
from Box2D import b2PolygonShape, b2Vec2, b2FixtureDef
from math import sin, cos, pi, sqrt
import cv2

"""
An agent that is always moving forward.
The action applies an angular velocity to it.

"""
class AgentForward:
    def __init__(
        self, simulator = None, x = 0, y = 0, 
        width = 1.0, height=2.0, 
        velocity = 1.0, numSteps=10,
        ang_vel_l=[-1.5, -0.75, 0, 0.75, 1.5] # Radians/seconds
        ):
        # define the current state of the agent
        # 0 - idle (can receive actions)
        # 1 - performing an action (can do nothing besides wait)
        self.state = 0

        # Actions and state keeping
        self.n_actions = len(ang_vel_l)
        self.ang_vel_l = ang_vel_l
        self.num_steps = numSteps
        self.velocity = velocity
        self.cur_action_steps = 0

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
        # Circumcircle:
        self.agent_radius = sqrt((width**2)/4 + (height**2)/4)
        self.agent_shape = b2PolygonShape(box=(width/2, height/2))
        self.agent_fixture_def = b2FixtureDef(shape=self.agent_shape, density=1, friction=0.0)
        self.agent_rigid_body.CreateFixture(self.agent_fixture_def)
        self.last_agent_pos = b2Vec2(x,y)


    def UpdateLastPos(self):
        self.last_agent_pos = self.agent_rigid_body.position.copy()

    def Dist(self, other):
        return (self.agent_rigid_body.position - other).length

    def GetPositionAsList(self):
        return (self.agent_rigid_body.position[0], self.agent_rigid_body.position[1])

    def GetRandomValidAction(self):
        # actions is the full action array
        actions = np.arange(0, self.n_actions, 1, dtype=np.int32)

        # it should be further filtered to encompass only
        # valid actions acording to a world view
        dice_pos = np.random.randint(0, len(actions))

        return actions[dice_pos]

    def PerformAction(self, action : int):
        self.agent_rigid_body.angularVelocity = self.ang_vel_l[action]
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
            agl = self.agent_rigid_body.angle
            dir = b2Vec2(cos(agl), sin(agl))
            self.agent_rigid_body.linearVelocity = dir * self.velocity

            self.cur_action_steps += 1
            if(self.cur_action_steps >= self.num_steps):
                self.state = 0


    # Draw functions
    def worldToScreen(self, position, pixels_per_meter):
        return (int(position[0] * pixels_per_meter), int(position[1] * pixels_per_meter))

    def Draw(self, pixels_per_meter, image, color, thickness):
        vertices = [(self.agent_rigid_body.transform * v) for v in self.agent_rigid_body.fixtures[0].shape.vertices]
        vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
        cv2.fillPoly(image, [np.array(vertices)], color)