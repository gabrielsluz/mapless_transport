from Box2D import (
    b2World, b2ContactListener, b2QueryCallback, 
    b2RayCastCallback, b2AABB, b2Vec2, b2_pi
)

import numpy as np
import cv2
import random
import dataclasses

from research_envs.b2PushWorld.Obstacle import CircleObs, RectangleObs, PolygonalObs
from research_envs.b2PushWorld.Object import CircleObj, RectangleObj, PolygonalObj
from research_envs.b2PushWorld.AgentForward import AgentForward


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
        if 'object' in type_l and 'obstacle' in type_l:
            self.simulator.object_collided += 1

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
    AGENT = 3
    
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
        if self.fixture.body.userData['type'] != 'obstacle':
            return -1
        return fraction


@dataclasses.dataclass
class TransportationWorldConfig:
    # ----------- World configuration -----------
    width: float = 50
    height: float = 50
    pixels_per_meter: int = 20
    # Obstacles
    obstacle_l: list = dataclasses.field(default_factory=list)
    # Objects
    object_l: list = dataclasses.field(default_factory=lambda: [{'name':'Circle', 'radius':4.0}])
    max_obj_dist: float = 14.0
    # Laser
    ang_min: float = -b2_pi # start angle of the scan [rad]
    ang_max: float = b2_pi # end angle of the scan [rad]
    n_rays: int = 16
    range_max: float = 4.0 # maximum range of the sensor [m]
    # Agent
    agent_type: str = 'forward'
    agent_width: float = 1.0
    agent_height: float = 2.0
    action_step_len: int = 10
    action_velocity: float = 2.0
    action_l: list = dataclasses.field(default_factory=lambda:[-1.5, -0.75, 0, 0.75, 1.5])
    # Goal
    goal_tolerance: float = 2.0



class TransportationWorld:
    def __init__(self, config: TransportationWorldConfig):
        # ----------- World configuration -----------
        self.world = b2World(gravity=(0, 0.0), doSleep=False)
        # the timestep is used to simulate discrete steps through the
        # engine's integrator and it is calculated in seconds
        self.timestep = 1.0 / 60.0
        # velocity and position iterations are used by the constraint solver
        self.vel_iterator = 6
        self.pos_iterator = 2
        # World dimensions in meters
        self.width = config.width
        self.height = config.height

        # Obstacles
        self.obstacle_l = []
        for obj_spec in config.obstacle_l:
            if obj_spec['name'] == 'Circle':
                self.obstacle_l.append(
                    CircleObs(
                        simulator=self, x=obj_spec['pos'][0], y=obj_spec['pos'][1], 
                        radius=obj_spec['radius']))
            elif obj_spec['name'] == 'Rectangle':
                self.obstacle_l.append(
                    RectangleObs(
                        simulator=self, x=obj_spec['pos'][0], y=obj_spec['pos'][1], 
                        height=obj_spec['height'], width=obj_spec['width']))
            elif obj_spec['name'] == 'Polygon':
                self.obstacle_l.append(
                    PolygonalObs(
                        simulator=self, x=obj_spec['pos'][0], y=obj_spec['pos'][1],
                        vertices=obj_spec['vertices']))
            else:
                raise ValueError('Unknown object type')

        # Objects
        self.obj_l = []
        for obj_spec in config.object_l:
            if obj_spec['name'] == 'Circle':
                self.obj_l.append(
                    CircleObj(simulator=self, radius=obj_spec['radius']))
            elif obj_spec['name'] == 'Rectangle':
                self.obj_l.append(
                    RectangleObj(simulator=self, height=obj_spec['height'], width=obj_spec['width']))
            elif obj_spec['name'] == 'Polygon':
                self.obj_l.append(
                    PolygonalObj(simulator=self, vertices=obj_spec['vertices']))
            else:
                raise ValueError('Unknown object type')
        self.obj = self.obj_l[random.randrange(0, len(self.obj_l))]
        self.max_obj_dist = config.max_obj_dist

        # Agent
        assert config.agent_type in ['forward']
        if config.agent_type == 'forward':
            self.agent = AgentForward(
                simulator=self, x=30, y=25,
                width=config.agent_width, height=config.agent_height,
                velocity=config.action_velocity, numSteps=config.action_step_len,
                ang_vel_l=config.action_l
            )

        # Goal
        self.goal = b2Vec2(0,0)
        self.goal_tolerance = config.goal_tolerance

        # Collisions
        self.contactListener = NavContactListener(simulator=self)
        self.world.contactListener = self.contactListener
        self.agent_collided = 0

        # Laser
        self.ang_min = config.ang_min
        self.ang_max = config.ang_max
        self.n_rays = config.n_rays
        self.range_max = config.range_max

        # ----------- Draw configuration -----------
        self.pixels_per_meter = config.pixels_per_meter
        self.screen_width = int(self.width * self.pixels_per_meter)
        self.screen_height = int(self.height * self.pixels_per_meter)

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

    def did_object_collide(self):
        return self.object_collided > 0

    def did_object_reach_goal(self):
        return (self.goal - self.obj.obj_rigid_body.position).length < self.goal_tolerance
    
    def agent_to_goal_vector(self):
        return self.goal - self.agent.agent_rigid_body.position
    
    def agent_to_goal_local_vector(self):
        return self.agent.agent_rigid_body.GetLocalPoint(self.goal)
    
    def agent_to_object_vector(self):
        return self.obj.obj_rigid_body.position - self.agent.agent_rigid_body.position
    
    def agent_to_object_local_vector(self):
        return self.agent.agent_rigid_body.GetLocalPoint(self.obj.obj_rigid_body.position)
    
    def object_to_goal_vector(self):
        return self.goal - self.obj.obj_rigid_body.position

    def gen_non_overlapping_position(self, radius):
        callback = CheckOverlapCallback()
        for _ in range(300):
            callback.reset()
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            aabb = b2AABB(lowerBound=(x-radius, y-radius), upperBound=(x+radius, y+radius))
            self.world.QueryAABB(callback, aabb)
            if not callback.overlapped:
                return (x, y)
        return (x, y) # If max iterations reached, return last position

    def gen_non_overlapping_position_in_limit(self, radius, x_lim, y_lim):
        callback = CheckOverlapCallback()
        for _ in range(300):
            callback.reset()
            x = random.uniform(x_lim[0], x_lim[1])
            y = random.uniform(y_lim[0], y_lim[1])
            aabb = b2AABB(lowerBound=(x-radius, y-radius), upperBound=(x+radius, y+radius))
            self.world.QueryAABB(callback, aabb)
            if not callback.overlapped:
                return (x, y)
        return (x, y) # If max iterations reached, return last position

    def reset(self):
        self.agent_collided = 0
        self.object_collided = 0

        self.obj = self.obj_l[random.randrange(0, len(self.obj_l))]
        self.obj.obj_rigid_body.position = self.gen_non_overlapping_position(
            1.2*(self.obj.obj_radius + 2*self.agent.agent_radius))
        self.obj.obj_rigid_body.angle = random.uniform(0, 2*np.pi)

        x_lim = [
            self.obj.obj_rigid_body.position.x - self.max_obj_dist*0.7071,
            self.obj.obj_rigid_body.position.x + self.max_obj_dist*0.7071
        ]
        y_lim = [
            self.obj.obj_rigid_body.position.y - self.max_obj_dist*0.7071,
            self.obj.obj_rigid_body.position.y + self.max_obj_dist*0.7071
        ]
        self.agent.agent_rigid_body.position = self.gen_non_overlapping_position_in_limit(
            self.agent.agent_radius*1.2, x_lim, y_lim)
        self.agent.agent_rigid_body.angle = random.uniform(0, 2*np.pi)

        sampled_pos = self.gen_non_overlapping_position(self.goal_tolerance+self.obj.obj_radius)
        self.goal.x = sampled_pos[0]
        self.goal.y = sampled_pos[1]

    # ----------- Laser Sensor -----------
    def get_laser_readings(self):
        ang_inc = (self.ang_max - self.ang_min) / self.n_rays

        range_l = []
        type_l = []
        point_l = []
        point1 = self.agent.agent_rigid_body.position
        agent_ang = self.agent.agent_rigid_body.angle

        ray_ang = self.ang_min
        for _ in range(self.n_rays):
            ang = ray_ang + agent_ang
            ray_ang += ang_inc
            point2 = point1 + self.range_max * b2Vec2(np.cos(ang), np.sin(ang))
            callback = RayCastClosestCallback()
            self.world.RayCast(callback, point1, point2)

            if callback.hit:
                body_type = callback.fixture.body.userData['type']
                if body_type == 'obstacle':
                    range_l.append((point1 - callback.point).length)
                    type_l.append(LaserHit.OBSTACLE)
                    point_l.append(callback.point)
                else:
                    range_l.append(self.range_max)#range_l.append(-1)
                    type_l.append(LaserHit.INVALID)
                    point_l.append(point2)
            else:
                range_l.append(self.range_max)#range_l.append(-1)
                type_l.append(LaserHit.INVALID)
                point_l.append(point2)
        return range_l, type_l, point_l

    def get_laser_readings_from_point(self, point1):
        ang_inc = (self.ang_max - self.ang_min) / self.n_rays

        range_l = []
        type_l = []
        point_l = []
        agent_ang = self.agent.agent_rigid_body.angle

        ray_ang = self.ang_min
        for _ in range(self.n_rays):
            ang = ray_ang + agent_ang
            ray_ang += ang_inc
            point2 = point1 + self.range_max * b2Vec2(np.cos(ang), np.sin(ang))
            callback = RayCastClosestCallback()
            self.world.RayCast(callback, point1, point2)

            if callback.hit:
                body_type = callback.fixture.body.userData['type']
                if body_type == 'obstacle':
                    range_l.append((point1 - callback.point).length)
                    type_l.append(LaserHit.OBSTACLE)
                    point_l.append(callback.point)
                else:
                    range_l.append(self.range_max)#range_l.append(-1)
                    type_l.append(LaserHit.INVALID)
                    point_l.append(point2)
            else:
                range_l.append(self.range_max)#range_l.append(-1)
                type_l.append(LaserHit.INVALID)
                point_l.append(point2)
        return range_l, type_l, point_l

    # ----------- Draw functions -----------
    def worldToScreen(self, position):
        return (int(position[0] * self.pixels_per_meter), int(position[1] * self.pixels_per_meter))

    def drawArrow(self, image, world_pos, angle, len, color):
        start_pos = self.worldToScreen(world_pos)
        end_pos = (world_pos[0] + len * np.cos(angle), world_pos[1] + len * np.sin(angle))
        end_pos = self.worldToScreen(end_pos)
        cv2.arrowedLine(image, start_pos, end_pos, color, thickness=6)

    def drawToBuffer(self):
        # clear previous buffer
        screen = np.ones(shape=(self.screen_height, self.screen_width, 3), dtype=np.float32)
        # Draw obstacles
        for obs in self.obstacle_l:
            obs.Draw(self.pixels_per_meter, screen, (0.5, 0.5, 0.5), -1)
        # Draw the object
        if self.object_collided == 0:
            self.obj.Draw(self.pixels_per_meter, screen, (0, 0, 1, 0), -1)
            self.drawArrow(screen, self.obj.GetPositionAsList() , self.obj.obj_rigid_body.angle, 10, (0, 0, 1, 0))
        else:
            self.obj.Draw(self.pixels_per_meter, screen, (1, 0, 1, 0), -1)
            self.drawArrow(screen, self.obj.GetPositionAsList() , self.obj.obj_rigid_body.angle, 10, (1, 0, 1, 0))
        # Draw max object dist
        screen_pos = self.worldToScreen(self.obj.GetPositionAsList())
        cv2.circle(screen, screen_pos, int(self.max_obj_dist*self.pixels_per_meter), (0, 0.5, 0.5, 0), thickness=4, lineType=4)
        # Draw goal
        screen_pos = self.worldToScreen(self.goal)
        cv2.circle(screen, screen_pos, int(self.goal_tolerance*self.pixels_per_meter), (0, 1, 0, 0), -1)
        # Draw agent
        screen_pos = self.worldToScreen(self.agent.GetPositionAsList())
        if self.agent_collided == 0:
            # self.agent.Draw(self.pixels_per_meter, screen, (1, 0, 0, 0), -1)
            cv2.circle(screen, screen_pos, int(self.agent.agent_radius*self.pixels_per_meter), (1, 0, 0, 0), -1)
        else:
            # self.agent.Draw(self.pixels_per_meter, screen, (0, 0, 1, 0), -1)
            cv2.circle(screen, screen_pos, int(self.agent.agent_radius*self.pixels_per_meter), (0, 0, 1, 0), -1)
        return screen

    def drawToBufferWithLaser(self):
        screen = self.drawToBuffer()
        _, _, laser_point_l = self.get_laser_readings()
        screen_pos = self.worldToScreen(self.agent.GetPositionAsList())
        for point in laser_point_l:
            # if point is not None:
            screen_point = self.worldToScreen(point)
            cv2.line(screen, screen_pos, screen_point, (0, 0, 1), 1)
        return screen

    def drawToBufferObservation(self):
        # clear previous buffer
        screen = np.ones(shape=(self.screen_height, self.screen_width, 3), dtype=np.float32)
        # Draw goal
        screen_pos = self.worldToScreen(self.goal)
        cv2.circle(screen, screen_pos, int(self.goal_tolerance*self.pixels_per_meter), (0, 1, 0, 0), -1)
        # Lasers
        _, _, laser_point_l = self.get_laser_readings()
        screen_pos = self.worldToScreen(self.agent.GetPositionAsList())
        for point in laser_point_l:
            # if point is not None:
            screen_point = self.worldToScreen(point)
            cv2.line(screen, screen_pos, screen_point, (0, 0, 1), 1)
        return screen

