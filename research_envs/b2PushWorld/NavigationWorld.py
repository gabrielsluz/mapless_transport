from Box2D import b2World, b2ContactListener, b2QueryCallback, b2AABB, b2Vec2

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
        self.screen = np.zeros(shape=(self.screen_height, self.screen_width), dtype=np.float32)

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

    # ----------- Draw functions -----------

    def worldToScreen(self, position):
        return (int(position[0] * self.pixels_per_meter), int(position[1] * self.pixels_per_meter))

    def drawToBuffer(self):
        # clear previous buffer
        self.screen = np.ones(shape=(self.screen_height, self.screen_width, 3), dtype=np.float32)
        # Draw obstacles
        for obs in self.obstacle_l:
            obs.Draw(self.pixels_per_meter, self.screen, (0.5, 0.5, 0.5), -1)
        # Draw goal
        screen_pos = self.worldToScreen(self.goal)
        cv2.circle(self.screen, screen_pos, int(self.goal_tolerance*self.pixels_per_meter), (0, 1, 0, 0), -1)
        # Draw agent
        screen_pos = self.worldToScreen(self.agent.GetPositionAsList())
        if self.agent_collided == 0:
            cv2.circle(self.screen, screen_pos, int(self.agent.agent_radius*self.pixels_per_meter), (1, 0, 0, 0), -1)
        else:
            cv2.circle(self.screen, screen_pos, int(self.agent.agent_radius*self.pixels_per_meter), (0, 0, 1, 0), -1)
        return self.screen