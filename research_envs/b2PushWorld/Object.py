from Box2D import b2CircleShape, b2PolygonShape, b2Transform, b2FixtureDef, b2Vec2, b2WeldJointDef, b2WeldJoint
import cv2
import math
import numpy as np

from research_envs.b2PushWorld.utils.poly_decomp import polygonDecomp

import shapely

class Object:
    # Inheritance: __init__ from derivated class must fill:
    #   obj_type, and obj_shape
    # Must implement the methods: __init__, Draw, DrawInPos
    def __init__(self, simulator = None, x = 0, y = 0):
        # ----------- Object configuration -----------
        self.obj_rigid_body = simulator.world.CreateDynamicBody(
            position=(x,y),
            linearDamping = 2.0,
            angularDamping = 5.0        
        )
        self.obj_fixture_def = b2FixtureDef(shape=self.obj_shape, density=1, friction=0.3)
        self.obj_rigid_body.CreateFixture(self.obj_fixture_def)
        self.obj_rigid_body.userData = {'type': 'object'}
        
        # the previous state should be stored inside the environment since
        # it depends on the execution of a full action step until time t+1
        self.last_obj_pos = b2Vec2(x,y)

    def UpdateLastPos(self):
        self.last_obj_pos = self.obj_rigid_body.position.copy()

    def Update(self):
        self.obj_rigid_body.linearVelocity = self.obj_rigid_body.linearVelocity * 0.98

    def GetPositionAsList(self):
        return (self.obj_rigid_body.position[0], self.obj_rigid_body.position[1])

    def worldToScreen(self, position, pixels_per_meter):
        return (int(position[0] * pixels_per_meter), int(position[1] * pixels_per_meter))


class CircleObj(Object):
    def __init__(self, simulator = None, x = 0, y = 0, radius=1.0):
        self.obj_type = 'Circle'
        self.obj_radius = radius
        self.obj_shape = b2CircleShape(radius=self.obj_radius)
        super().__init__(simulator, x, y)

    def Draw(self, pixels_per_meter, image, color, thickness):
        position = self.GetPositionAsList()
        screen_pos = self.worldToScreen(position, pixels_per_meter)
        cv2.circle(image, screen_pos, int(self.obj_radius*pixels_per_meter), color, thickness)

    def DrawInPos(self, screen_pos, pixels_per_meter, image, color, thickness):
        cv2.circle(image, screen_pos, int(self.obj_radius*pixels_per_meter), color, thickness)
    
    def DrawInPose(self, world_pos, angle, pixels_per_meter, image, color, thickness):
        position = [world_pos.x, world_pos.y]
        screen_pos = self.worldToScreen(position, pixels_per_meter)
        cv2.circle(image, screen_pos, int(self.obj_radius*pixels_per_meter), color, thickness)


class RectangleObj(Object):
    def __init__(self, simulator = None, x = 0, y = 0, height=1.0, width=1.0):
        self.obj_type = 'Rectangle'
        self.height = height
        self.width = width
        self.obj_shape = b2PolygonShape(box=(width/2, height/2))
        self.obj_radius = math.sqrt((width/2)**2 + (height/2)**2)
        super().__init__(simulator, x, y)

    def Draw(self, pixels_per_meter, image, color, thickness):
        vertices = [(self.obj_rigid_body.transform * v) for v in self.obj_rigid_body.fixtures[0].shape.vertices]
        vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
        cv2.fillPoly(image, [np.array(vertices)], color)

    def DrawInPos(self, screen_pos, pixels_per_meter, image, color, thickness):
        # Only rotate
        vertices = [(self.obj_rigid_body.transform.q * v) for v in self.obj_rigid_body.fixtures[0].shape.vertices]
        vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
        # Translate
        vertices = [(v[0]+screen_pos[0], v[1]+screen_pos[1]) for v in vertices]
        cv2.fillPoly(image, [np.array(vertices)], color)

    def DrawInPose(self, world_pos, angle, pixels_per_meter, image, color, thickness):
        transform_matrix = b2Transform()
        transform_matrix.SetIdentity()
        transform_matrix.Set(world_pos, angle)
        vertices = [(transform_matrix * v) for v in self.obj_rigid_body.fixtures[0].shape.vertices]
        vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
        cv2.fillPoly(image, [np.array(vertices)], color)


class PolygonalObj(Object):
    def __init__(self, simulator = None, x = 0, y = 0, vertices=[]):
        # Vertices in counterclockwise
        # Adjusts such that centroid is in the center (0,0)
        self.obj_type = 'Polygon'
        aux_obj_shape = b2PolygonShape(vertices=vertices)
        centroid = aux_obj_shape.centroid
        vertices = [(v[0]-centroid[0], v[1]-centroid[1]) for v in vertices]
        self.obj_shape = b2PolygonShape(vertices=vertices)
        # Compute radius
        max_d = -1
        for v in vertices:
            d = math.sqrt(v[0]**2 + v[1]**2)
            if d > max_d: max_d = d
        self.obj_radius = max_d

        super().__init__(simulator, x, y)

    def Draw(self, pixels_per_meter, image, color, thickness):
        vertices = [(self.obj_rigid_body.transform * v) for v in self.obj_rigid_body.fixtures[0].shape.vertices]
        vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
        cv2.fillPoly(image, [np.array(vertices)], color)

    def DrawInPos(self, screen_pos, pixels_per_meter, image, color, thickness):
        # Only rotate
        vertices = [(self.obj_rigid_body.transform.q * v) for v in self.obj_rigid_body.fixtures[0].shape.vertices]
        vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
        # Translate
        vertices = [(v[0]+screen_pos[0], v[1]+screen_pos[1]) for v in vertices]
        cv2.fillPoly(image, [np.array(vertices)], color)

    def DrawInPose(self, world_pos, angle, pixels_per_meter, image, color, thickness):
        transform_matrix = b2Transform()
        transform_matrix.SetIdentity()
        transform_matrix.Set(world_pos, angle)
        vertices = [(transform_matrix * v) for v in self.obj_rigid_body.fixtures[0].shape.vertices]
        vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
        cv2.fillPoly(image, [np.array(vertices)], color)


class MultiPolygonsObj:
    # To partition a simple polygon into convex parts use: https://github.com/ivanfratric/polypartition/tree/master
    def __init__(self,  poly_vertices_l, simulator = None, x = 0, y = 0):
        # poly_vertices_l: [[[0, 0], [1, 1], [0, 2]], [[0, 0], [-2, 0], [-1, -1]]]
        # Vertices in counterclockwise
        # Adjusts such that centroid is in the center (0,0)
        self.obj_type = 'MultiPolygons'

        # Centralize
        aux_l = [
            (vertices, []) for vertices in poly_vertices_l
        ]
        multi_poly = shapely.MultiPolygon(aux_l)
        c = shapely.centroid(multi_poly)
        c_x = shapely.get_x(c)
        c_y = shapely.get_y(c)

        aux_l = []
        for vertices in poly_vertices_l:
            aux_l.append(
                [[v[0] - c_x, v[1] - c_y] for v in vertices]
            )
        poly_vertices_l = aux_l

        # Compute radius
        max_d = -1
        for vertices in poly_vertices_l:
            for v in vertices:
                d = math.sqrt(v[0]**2 + v[1]**2)
                if d > max_d: max_d = d
        self.obj_radius = max_d

        body = simulator.world.CreateDynamicBody(
            position=(x, y),
            linearDamping = 2.0,
            angularDamping = 5.0        
        )
        body.userData = {'type': 'object'}

        for vertices in poly_vertices_l:
            body.CreateFixture(
                b2FixtureDef(
                    shape=b2PolygonShape(vertices=vertices), 
                    density=1, friction=0.3))

        print("CoM:", body.localCenter) # Should be very close to (0, 0)

        self.obj_rigid_body = body
        self.last_obj_pos = self.obj_rigid_body.position

    def UpdateLastPos(self):
        self.last_obj_pos = self.obj_rigid_body.position.copy()

    def Update(self):
        self.obj_rigid_body.linearVelocity = self.obj_rigid_body.linearVelocity * 0.98

    def GetPositionAsList(self):
        return (self.obj_rigid_body.position[0], self.obj_rigid_body.position[1])

    def worldToScreen(self, position, pixels_per_meter):
        return (int(position[0] * pixels_per_meter), int(position[1] * pixels_per_meter))


    def Draw(self, pixels_per_meter, image, color, thickness):
        body = self.obj_rigid_body
        for f_i in range(len(body.fixtures)):
            vertices = [(body.transform * v) for v in body.fixtures[f_i].shape.vertices]
            vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
            cv2.fillPoly(image, [np.array(vertices)], color)
        
        # Aux:
        # v = body.worldCenter
        # v = self.worldToScreen(v, pixels_per_meter)
        # cv2.circle(image, v, 10, (1, 0, 0, 0), 5)

    def DrawInPos(self, screen_pos, pixels_per_meter, image, color, thickness):
        body = self.obj_rigid_body
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
        body = self.obj_rigid_body
        for f_i in range(len(body.fixtures)):
            vertices = [(transform_matrix * v) for v in body.fixtures[f_i].shape.vertices]
            vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
            cv2.fillPoly(image, [np.array(vertices)], color)

# Dev

class ConcavePolygonObj:
    def _compute_centroid(self, vertices):
        # https://paulbourke.net/geometry/polygonmesh/
        # Compute the area
        v = vertices + [vertices[0]]
        A = 0
        for i in range(len(vertices)):
            A += v[i][0] * v[i+1][1] - v[i+1][0] * v[i][1]
        A = A / 2

        c_x = 0
        for i in range(len(vertices)):
            c_x += (v[i][0] + v[i+1][0]) * (v[i][0] * v[i+1][1] - v[i+1][0] * v[i][1])
        c_x = c_x / (6 * A)

        c_y = 0
        for i in range(len(vertices)):
            c_y += (v[i][1] + v[i+1][1]) * (v[i][0] * v[i+1][1] - v[i+1][0] * v[i][1])
        c_y = c_y / (6 * A)

        return c_x, c_y

    def __init__(self, simulator = None, x = 0, y = 0, vertices=[]):
        # Vertices in counterclockwise => list of lists 
        # Decomposes the shape into a set of convex polygons.
        # Adjusts such that centroid is in the center (0,0)
        self.obj_type = 'ConcavePolygon'

        # Centralize
        c_x, c_y = self._compute_centroid(vertices)
        vertices = [[v[0] - c_x, v[1] - c_y] for v in vertices]

        # Compute radius
        max_d = -1
        for v in vertices:
            d = math.sqrt(v[0]**2 + v[1]**2)
            if d > max_d: max_d = d
        self.obj_radius = max_d

        # Decompose into a set of convex polygons
        poly_l = polygonDecomp(vertices) # Not working well

        body = simulator.world.CreateDynamicBody(
            position=(x, y),
            linearDamping = 2.0,
            angularDamping = 5.0        
        )
        body.userData = {'type': 'object'}

        for p_i in range(len(poly_l)):
            body.CreateFixture(
                b2FixtureDef(
                    shape=b2PolygonShape(vertices=poly_l[p_i]), 
                    density=1, friction=0.3))

        print("CoM:", body.localCenter) # Should be very close to (0, 0)
        print(poly_l)

        self.obj_rigid_body = body
        self.last_obj_pos = self.obj_rigid_body.position

    def UpdateLastPos(self):
        self.last_obj_pos = self.obj_rigid_body.position.copy()

    def Update(self):
        self.obj_rigid_body.linearVelocity = self.obj_rigid_body.linearVelocity * 0.98

    def GetPositionAsList(self):
        return (self.obj_rigid_body.position[0], self.obj_rigid_body.position[1])

    def worldToScreen(self, position, pixels_per_meter):
        return (int(position[0] * pixels_per_meter), int(position[1] * pixels_per_meter))


    def Draw(self, pixels_per_meter, image, color, thickness):
        body = self.obj_rigid_body
        for f_i in range(len(body.fixtures)):
            vertices = [(body.transform * v) for v in body.fixtures[f_i].shape.vertices]
            vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
            cv2.fillPoly(image, [np.array(vertices)], color)
        
        # Aux:
        v = body.worldCenter
        v = self.worldToScreen(v, pixels_per_meter)
        cv2.circle(image, v, 10, (1, 0, 0, 0), 5)

    def DrawInPos(self, screen_pos, pixels_per_meter, image, color, thickness):
        body = self.obj_rigid_body
        for f_i in range(len(body.fixtures)):
            # Only rotate
            vertices = [(body.transform.q * v) for v in body.fixtures[f_i].shape.vertices]
            vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
            # Translate
            vertices = [(v[0]+screen_pos[0], v[1]+screen_pos[1]) for v in vertices]
            cv2.fillPoly(image, [np.array(vertices)], color)


# class LObj:
#     def _compute_centroid(self, vertices):
#         # https://paulbourke.net/geometry/polygonmesh/
#         # Compute the area
#         v = vertices + [vertices[0]]
#         A = 0
#         for i in range(len(vertices)):
#             A += v[i][0] * v[i+1][1] - v[i+1][0] * v[i][1]
#         A = A / 2

#         c_x = 0
#         for i in range(len(vertices)):
#             c_x += (v[i][0] + v[i+1][0]) * (v[i][0] * v[i+1][1] - v[i+1][0] * v[i][1])
#         c_x = c_x / (6 * A)

#         c_y = 0
#         for i in range(len(vertices)):
#             c_y += (v[i][1] + v[i+1][1]) * (v[i][0] * v[i+1][1] - v[i+1][0] * v[i][1])
#         c_y = c_y / (6 * A)

#         return c_x, c_y

#     def __init__(self, simulator = None):
#         # Vertices in counterclockwise
#         # Adjusts such that centroid is in the center (0,0)
#         self.obj_type = 'L'
#         h0 = 4
#         w0 = 4
#         h1 = 4
#         w1 = 8

#         body = simulator.world.CreateDynamicBody(
#             position=(25, 25),
#             linearDamping = 2.0,
#             angularDamping = 5.0        
#         )
#         body.userData = {'type': 'object'}

#         # body.CreateFixture(
#         #     b2FixtureDef(
#         #         shape=b2PolygonShape(box=(w0/2, h0/2)), 
#         #         density=1, friction=0.3))
#         # print(body.fixtures[0].shape.vertices)
#         #v = [(-2.0, -2.0 + 4.0), (2.0, -2.0 + 4.0), (2.0, 2.0 + 4.0), (-2.0, 2.0 + 4.0)]
#         v = [(2.0, 2.0), (4.0, 2.0), (4.0, 6.0), (2.0, 6.0)]
#         body.CreateFixture(
#             b2FixtureDef(
#                 shape=b2PolygonShape(vertices=v), 
#                 density=1, friction=0.3))
#         body.CreateFixture(
#             b2FixtureDef(
#                 shape=b2PolygonShape(box=(w1/2, h1/2)), 
#                 density=1, friction=0.3))

#         # v = [(-4, -2), (4, -2), (4, 6), (2, 6), (2, 2), (-4, 2)]
#         v = [[-4, -2], [4, -2], [4, 6], [2, 6], [2, 2], [-4, 2]]
#         # v.reverse()
#         print('Centroid:', self._compute_centroid(v))
#         print('CoM:', body.localCenter)
        
#         print(polygonDecomp(v))

#         print(body.fixtures[0].shape.vertices)
#         print(body.fixtures[1].shape.vertices)

#         self.obj_rigid_body = body
#         self.last_obj_pos = self.obj_rigid_body.position

#         # Radius
#         self.obj_radius = math.sqrt((w1/2)**2 + (h0 + h1/2)**2)


#     def UpdateLastPos(self):
#         self.last_obj_pos = self.obj_rigid_body.position.copy()

#     def Update(self):
#         self.obj_rigid_body.linearVelocity = self.obj_rigid_body.linearVelocity * 0.98

#     def GetPositionAsList(self):
#         return (self.obj_rigid_body.position[0], self.obj_rigid_body.position[1])

#     def worldToScreen(self, position, pixels_per_meter):
#         return (int(position[0] * pixels_per_meter), int(position[1] * pixels_per_meter))


#     def Draw(self, pixels_per_meter, image, color, thickness):
#         body = self.obj_rigid_body
#         for f_i in range(len(body.fixtures)):
#             vertices = [(body.transform * v) for v in body.fixtures[f_i].shape.vertices]
#             vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
#             cv2.fillPoly(image, [np.array(vertices)], color)
        
#         # Aux:
#         v = body.worldCenter
#         v = self.worldToScreen(v, pixels_per_meter)
#         cv2.circle(image, v, 10, (1, 0, 0, 0), 5)

#     def DrawInPos(self, screen_pos, pixels_per_meter, image, color, thickness):
#         body = self.obj_rigid_body
#         for f_i in range(len(body.fixtures)):
#             # Only rotate
#             vertices = [(body.transform.q * v) for v in body.fixtures[f_i].shape.vertices]
#             vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
#             # Translate
#             vertices = [(v[0]+screen_pos[0], v[1]+screen_pos[1]) for v in vertices]
#             cv2.fillPoly(image, [np.array(vertices)], color)

# class LObj:
#     def __init__(self, simulator = None):
#         # Vertices in counterclockwise
#         # Adjusts such that centroid is in the center (0,0)
#         self.obj_type = 'L'
#         h0 = 4
#         w0 = 4
#         h1 = 4
#         w1 = 8
#         # Create the main body
#         shape = b2PolygonShape(box=(w0/2, h0/2))
#         rigid_body0 = simulator.world.CreateDynamicBody(
#             position=(25 + w0/2, 25 + h1 + h0/2),
#             linearDamping = 2.0,
#             angularDamping = 5.0        
#         )
#         rigid_body0.CreateFixture(
#             b2FixtureDef(shape=shape, density=1, friction=0.3))
#         rigid_body0.userData = {'type': 'object'}

#         # Create the second body
#         shape = b2PolygonShape(box=(w1/2, h1/2))
#         rigid_body1 = simulator.world.CreateDynamicBody(
#             position=(25 + w1/2, 25 + h1/2),
#             linearDamping = 2.0,
#             angularDamping = 5.0        
#         )
#         rigid_body1.CreateFixture(
#             b2FixtureDef(shape=shape, density=1, friction=0.3))
#         rigid_body1.userData = {'type': 'object'}

#         joint_def = b2WeldJointDef(
#             bodyA=rigid_body0, bodyB=rigid_body1, anchor=rigid_body0.position)
#         joint = simulator.world.CreateJoint(joint_def)


#         self.obj_rigid_body = rigid_body0
#         self.last_obj_pos = self.obj_rigid_body.position

#         self.body_l = [rigid_body0, rigid_body1]
#         # Radius
#         self.obj_radius = math.sqrt((w1/2)**2 + (h0 + h1/2)**2)


#     def UpdateLastPos(self):
#         self.last_obj_pos = self.obj_rigid_body.position.copy()

#     def Update(self):
#         self.obj_rigid_body.linearVelocity = self.obj_rigid_body.linearVelocity * 0.98

#     def GetPositionAsList(self):
#         return (self.obj_rigid_body.position[0], self.obj_rigid_body.position[1])

#     def worldToScreen(self, position, pixels_per_meter):
#         return (int(position[0] * pixels_per_meter), int(position[1] * pixels_per_meter))


#     def Draw(self, pixels_per_meter, image, color, thickness):
#         for body in self.body_l:
#             vertices = [(body.transform * v) for v in body.fixtures[0].shape.vertices]
#             vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
#             cv2.fillPoly(image, [np.array(vertices)], color)

#     def DrawInPos(self, screen_pos, pixels_per_meter, image, color, thickness):
#         for body in self.body_l:
#             # Only rotate
#             vertices = [(body.transform.q * v) for v in body.fixtures[0].shape.vertices]
#             vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
#             # Translate
#             vertices = [(v[0]+screen_pos[0], v[1]+screen_pos[1]) for v in vertices]
#             cv2.fillPoly(image, [np.array(vertices)], color)

        


    # def __init__(self, simulator = None, x = 0, y = 0, vertices=[]):
    #     # Vertices in counterclockwise
    #     # Adjusts such that centroid is in the center (0,0)
    #     self.obj_type = 'Polygon'
    #     aux_obj_shape = b2PolygonShape(vertices=vertices)
    #     centroid = aux_obj_shape.centroid
    #     vertices = [(v[0]-centroid[0], v[1]-centroid[1]) for v in vertices]
    #     self.obj_shape = b2PolygonShape(vertices=vertices)
    #     super().__init__(simulator, x, y)

    # def Draw(self, pixels_per_meter, image, color, thickness):
    #     vertices = [(self.obj_rigid_body.transform * v) for v in self.obj_rigid_body.fixtures[0].shape.vertices]
    #     vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
    #     cv2.fillPoly(image, [np.array(vertices)], color)

    # def DrawInPos(self, screen_pos, pixels_per_meter, image, color, thickness):
    #     # Only rotate
    #     vertices = [(self.obj_rigid_body.transform.q * v) for v in self.obj_rigid_body.fixtures[0].shape.vertices]
    #     vertices = [self.worldToScreen(v, pixels_per_meter) for v in vertices]
    #     # Translate
    #     vertices = [(v[0]+screen_pos[0], v[1]+screen_pos[1]) for v in vertices]
    #     cv2.fillPoly(image, [np.array(vertices)], color)