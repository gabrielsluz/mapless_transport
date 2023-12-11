import numpy as np
from Box2D import b2CircleShape, b2Vec2, b2FixtureDef
from math import sin, cos, pi

# Agent that take actions according to a potential field
class AgentDirection:
    def __init__(self, simulator = None, x = 0, y = 0, radius = 0.5, velocity = 1.0, maxForceLength = 5.0, minForceLength = 1.0):
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
        self.agent_rigid_body  = simulator.world.CreateDynamicBody(position=(x,y), fixedRotation=True)
        self.agent_rigid_body.userData = {'type': 'agent'} # For collision detection

        # ----------- Body configuration ------------
        self.agent_radius = radius
        self.agent_shape = b2CircleShape(radius=self.agent_radius)
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
        # spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        theta = 2*pi * action[0] 
        direction = np.array([cos(theta), sin(theta)])
        force = self.min_force_length + (self.max_force_length - self.min_force_length) * action[1]
        
        self.current_obj = self.agent_rigid_body.position + (force * direction)
        self.current_obj = b2Vec2(self.current_obj)

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