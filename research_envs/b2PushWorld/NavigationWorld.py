from Box2D import (
    b2World, b2ContactListener, b2QueryCallback, 
    b2RayCastCallback, b2AABB, b2Vec2, b2_pi
)

import numpy as np
import cv2
import random

from research_envs.b2PushWorld.Obstacle import CircleObs, RectangleObs
from research_envs.b2PushWorld.Agent import Agent

# ---------- For handling collisions ----------
"""
Detects when the agent collides with an obstacle.
"""
class NavContactListener(b2ContactListener):
    def __init__(self, simulator):
        super(NavContactListener, self).__init__()
        self.simulator = simulator

    def BeginContact(self, contact):
        super(NavContactListener, self).BeginContact(contact)
        type_l = [
            contact.fixtureA.body.userData['type'],
            contact.fixtureB.body.userData['type']
        ]
        if 'agent' in type_l and 'obstacle' in type_l:
            self.simulator.agent_collided += 1


# ---------- For reseting without overlapping ----------
class CheckOverlapCallback(b2QueryCallback):
    def __init__(self):
        super(CheckOverlapCallback, self).__init__()
        self.overlapped = False

    def ReportFixture(self, fixture):
        self.overlapped = True
        return False

    def reset(self):
        self.overlapped = False


# ---------- Laser rangefinder ----------
class LaserHit:
    INVALID = 0
    OBSTACLE = 1
    OBJECT = 2
    
class RayCastClosestCallback(b2RayCastCallback):
    """This callback finds the closest hit"""

    def __repr__(self):
        return 'Closest hit'

    def __init__(self, **kwargs):
        b2RayCastCallback.__init__(self, **kwargs)
        self.fixture = None
        self.hit = False

    def ReportFixture(self, fixture, point, normal, fraction):
        '''
        Called for each fixture found in the query. You control how the ray
        proceeds by returning a float that indicates the fractional length of
        the ray. By returning 0, you set the ray length to zero. By returning
        the current fraction, you proceed to find the closest point. By
        returning 1, you continue with the original ray clipping. By returning
        -1, you will filter out the current fixture (the ray will not hit it).
        '''
        self.hit = True
        self.fixture = fixture
        self.point = b2Vec2(point)
        # self.normal = b2Vec2(normal)
        # NOTE: You will get this error:
        #   "TypeError: Swig director type mismatch in output value of
        #    type 'float32'"
        # without returning a value
        return fraction


class NavigationWorld:
    def __init__(self):
        # ----------- World configuration -----------
        self.world = b2World(gravity=(0, 0.0), doSleep=False)
        # the timestep is used to simulate discrete steps through the
        # engine's integrator and it is calculated in seconds
        self.timestep = 1.0 / 60.0
        # velocity and position iterations are used by the constraint solver
        self.vel_iterator = 6
        self.pos_iterator = 2
        # World dimensions in meters
        self.width = 50
        self.height = 50

        # Obstacles
        self.obstacle_l = [
            CircleObs(simulator=self, x = 5, y = 5, radius=2.0),
            CircleObs(simulator=self, x = 35, y = 35, radius=2.0),
            RectangleObs(simulator=self, height=10, width=2, x=25, y=25),
        ]

        # Agent
        self.agent = Agent(
            simulator=self, x=30, y=25,
            radius=1.0,
            velocity=2.0, forceLength=2.0,
            totalDirections=8)

        # Goal
        self.goal = b2Vec2(0,0)
        self.goal_tolerance = 2.0

        # Collisions
        self.contactListener = NavContactListener(simulator=self)
        self.world.contactListener = self.contactListener
        self.agent_collided = 0

        # ----------- Draw configuration -----------
        self.pixels_per_meter = 20
        self.screen_width = self.width * self.pixels_per_meter
        self.screen_height = self.height * self.pixels_per_meter

        # Reset
        self.reset()

    def update(self):
        self.agent.Update()
        self.world.Step(
            timeStep=self.timestep,
            velocityIterations=self.vel_iterator,
            positionIterations=self.pos_iterator)
        self.world.ClearForces()

    def take_action(self, action):
        self.agent.PerformAction(action)
        while(self.agent.IsPerformingAction()):
            self.update()

    def did_agent_collide(self):
        return self.agent_collided > 0

    def did_agent_reach_goal(self):
        return (self.goal - self.agent.agent_rigid_body.position).length < self.goal_tolerance

    def gen_non_overlapping_position(self, radius):
        callback = CheckOverlapCallback()
        for _ in range(100):
            callback.reset()
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            aabb = b2AABB(lowerBound=(x-radius, y-radius), upperBound=(x+radius, y+radius))
            self.world.QueryAABB(callback, aabb)
            if not callback.overlapped:
                return (x, y)
        return (x, y) # If max iterations reached, return last position

    def reset(self):
        self.agent_collided = 0
        self.agent.agent_rigid_body.position = self.gen_non_overlapping_position(self.agent.agent_radius)
        sampled_pos = self.gen_non_overlapping_position(self.goal_tolerance)
        self.goal.x = sampled_pos[0]
        self.goal.y = sampled_pos[1]

    # ----------- Laser Sensor -----------
    def get_laser_readings(self):
        # Params:
        ang_min = -b2_pi # start angle of the scan [rad]
        ang_max = b2_pi # end angle of the scan [rad]
        rays = 16 # indirect parameter, not used in the code
        ang_inc = (ang_max - ang_min) / rays
        range_max = 4.0 # maximum range of the sensor [m]

        range_l = []
        type_l = []
        point_l = []
        point1 = self.agent.agent_rigid_body.position
        agent_ang = self.agent.agent_rigid_body.angle

        ray_ang = ang_min
        while ray_ang <= ang_max:
            ang = ray_ang + agent_ang
            ray_ang += ang_inc
            point2 = point1 + range_max * b2Vec2(np.cos(ang), np.sin(ang))
            callback = RayCastClosestCallback()
            self.world.RayCast(callback, point1, point2)

            if callback.hit:
                range_l.append((point1 - callback.point).length)
                type_l.append(LaserHit.OBSTACLE)
                point_l.append(callback.point)
            else:
                range_l.append(-1)
                type_l.append(LaserHit.INVALID)
                point_l.append(point2)
        return range_l, type_l, point_l

    # ----------- Draw functions -----------

    def worldToScreen(self, position):
        return (int(position[0] * self.pixels_per_meter), int(position[1] * self.pixels_per_meter))

    def drawToBuffer(self):
        # clear previous buffer
        screen = np.ones(shape=(self.screen_height, self.screen_width, 3), dtype=np.float32)
        # Draw obstacles
        for obs in self.obstacle_l:
            obs.Draw(self.pixels_per_meter, screen, (0.5, 0.5, 0.5), -1)
        # Draw goal
        screen_pos = self.worldToScreen(self.goal)
        cv2.circle(screen, screen_pos, int(self.goal_tolerance*self.pixels_per_meter), (0, 1, 0, 0), -1)
        # Draw agent
        screen_pos = self.worldToScreen(self.agent.GetPositionAsList())
        if self.agent_collided == 0:
            cv2.circle(screen, screen_pos, int(self.agent.agent_radius*self.pixels_per_meter), (1, 0, 0, 0), -1)
        else:
            cv2.circle(screen, screen_pos, int(self.agent.agent_radius*self.pixels_per_meter), (0, 0, 1, 0), -1)
        return screen

    def drawToBufferWithLaser(self, laser_point_l):
        screen = self.drawToBuffer()
        # Draw laser
        screen_pos = self.worldToScreen(self.agent.GetPositionAsList())
        for point in laser_point_l:
            # if point is not None:
            screen_point = self.worldToScreen(point)
            cv2.line(screen, screen_pos, screen_point, (0, 0, 1), 1)
        return screen

