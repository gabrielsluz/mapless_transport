# To execute from root folder
import sys
sys.path.append('.')

# from research_envs.path_planning.vis_graph import create_vis_g

from research_envs.envs.obstacle_repo import obstacle_l_dict

from shapely.geometry import Point, Polygon
import shapely
import numpy as np
from Box2D import b2PolygonShape

import matplotlib.pyplot as plt
from research_envs.path_planning.descartes_mod.patch import PolygonPatch
from research_envs.path_planning.vis_graph import create_vis_g, find_path_vis_g
from research_envs.path_planning.map_utils import plot_map_path

def plot_map(obs_set, x_rng, y_rng):
    """
    Mostra o mapa de polígonos Shapely
    """
    fig = plt.figure(figsize=(8,5), dpi=100)
    ax = fig.add_subplot(111, aspect='equal') 

    for obs in obs_set:
        print(obs)
        ax.add_patch(PolygonPatch(obs, facecolor='gray'))
        
    ax.set_xlim(x_rng[0], x_rng[1])
    ax.set_ylim(y_rng[0], y_rng[1])
    plt.show()

def plot_graph(obs_set, vis_g, x_rng, y_rng):
    """
    Mostra o mapa de polígonos Shapely
    """
    fig = plt.figure(figsize=(8,5), dpi=100)
    ax = fig.add_subplot(111, aspect='equal') 

    for obs in obs_set:
        print(obs)
        ax.add_patch(PolygonPatch(obs, facecolor='gray'))
    
    x_l = [vis_g.nodes[n_i]['coord'][0] for n_i in vis_g.nodes]
    y_l = [vis_g.nodes[n_i]['coord'][1] for n_i in vis_g.nodes]
    plt.scatter(x_l, y_l)
        
    ax.set_xlim(x_rng[0], x_rng[1])
    ax.set_ylim(y_rng[0], y_rng[1])
    plt.show()

# Create shapely map from description
obj_spec_l = obstacle_l_dict['sparse_2']
print(obj_spec_l)

shape_objs_l = []
for obj_spec in obj_spec_l:
    if obj_spec['name'] == 'Circle':
        shape_objs_l.append(
            Point(obj_spec['pos'][0],obj_spec['pos'][1]).buffer(obj_spec['radius'])
        )
    elif obj_spec['name'] == 'Rectangle':
        x, y = obj_spec['pos'][0], obj_spec['pos'][1]
        w, h = width=obj_spec['width'], obj_spec['height']
        shape_objs_l.append(
            Polygon(
                np.array([
                    [x - w/2, y + h/2],
                    [x - w/2, y - h/2],
                    [x + w/2, y - h/2],
                    [x + w/2, y + h/2],
                ])
            )
        )
    elif obj_spec['name'] == 'Polygon':
        x, y = obj_spec['pos'][0], obj_spec['pos'][1]
        vertices = obj_spec['vertices']

        aux_obj_shape = b2PolygonShape(vertices=vertices)
        centroid = aux_obj_shape.centroid
        vertices = [(v[0]-centroid[0] + x, v[1]-centroid[1] + y) for v in vertices]

        shape_objs_l.append(
            Polygon(np.array(vertices))
        )


objs_l_dilated = [
    shapely.buffer(o, 1) for o in shape_objs_l
]

plot_map(objs_l_dilated, (0, 50), (0, 50))

# r = 1
# orig_conf = np.array([0, 0, 0])
# dest_conf = np.array([50, 50, 0])

# vis_g = create_vis_g(shape_objs_l, objs_l_dilated, r)
# # plot_graph(shape_objs_l, vis_g, (0, 50), (0, 50))


# conf_path, path_corridors = find_path_vis_g(vis_g, shape_objs_l, orig_conf, dest_conf, r)

# plot_map_path(shape_objs_l, path_corridors, (0, 50), (0, 50))

# plot_map(objs_l_dilated, (0, 50), (0, 50))

# for obj_spec in config.obstacle_l:
#             if obj_spec['name'] == 'Circle':
#                 self.obstacle_l.append(
#                     CircleObs(
#                         simulator=self, x=obj_spec['pos'][0], y=obj_spec['pos'][1], 
#                         radius=obj_spec['radius']))
#             elif obj_spec['name'] == 'Rectangle':
#                 self.obstacle_l.append(
#                     RectangleObs(
#                         simulator=self, x=obj_spec['pos'][0], y=obj_spec['pos'][1], 
#                         height=obj_spec['height'], width=obj_spec['width']))
#             elif obj_spec['name'] == 'Polygon':
#                 self.obstacle_l.append(
#                     PolygonalObs(
#                         simulator=self, x=obj_spec['pos'][0], y=obj_spec['pos'][1],
#                         vertices=obj_spec['vertices']))
#             else:
#                 raise ValueError('Unknown object type')
