"""
Implementação do grafo de visibilidade
"""


import numpy as np
import math
import networkx as nx

from shapely.geometry import Polygon
from shapely.geometry import Point
import shapely

from research_envs.path_planning.path_utils import corridor_v_u, is_edge_valid

def create_vis_g(orig_objs_l, shape_objs_l, r):
    """
    Creates the Visibility Graph.
    r: robot radius
    """
    
    vertex_l = []
    for obs_with_holes in shape_objs_l:
        vertex_l += [
            np.array(c) for c in obs_with_holes.exterior.coords
        ]
    
    # Gera o grafo
    vis_g = nx.Graph()
    for i, v in enumerate(vertex_l):
        vis_g.add_node(i, coord=v)
    
    # Adiciona as arestas
    for i in range(len(vertex_l)):
        for j in range(i+1, len(vertex_l)):
            v = vertex_l[i]
            u = vertex_l[j]
            if is_edge_valid(v, u, orig_objs_l, r):
                edge_w = np.linalg.norm(u-v)
                vis_g.add_edge(i, j, weight=edge_w)
    return vis_g

def find_path_vis_g(vis_g, shape_objs_l, orig_conf, dest_conf, r):
    """
    Entrada:
        vis_g: Grafo de visibilidade
        shape_objs_l: Lista de polígonos Shapely
        orig_conf, dest_conf => configurações (x, y, theta)
        r: raio do circulo que representa o robô
    Retorna:
        conf_path: sequência de configurações da orig até dest_conf. O theta de todas
            as confs do path são iguais ao da dest_conf, exceto a da origem
        path_corridors: sequência de polígonos Shapely que mostram o caminho.
    """
    pos_orig = orig_conf[:2]
    pos_dest = dest_conf[:2]
    # Cria uma cópia do vis_g e adiciona os pontos
    plan_g = nx.Graph(vis_g)
    plan_g.add_node('orig', coord=pos_orig)
    for n_i in vis_g.nodes:
        v = pos_orig
        u = plan_g.nodes[n_i]['coord']
        if is_edge_valid(v, u, shape_objs_l, r):
            edge_w = np.linalg.norm(u-v)
            plan_g.add_edge('orig', n_i, weight=edge_w)
    
    plan_g.add_node('dest', coord=pos_dest)
    for n_i in plan_g.nodes:
        if n_i == 'dest':
            continue
        v = pos_dest
        u = plan_g.nodes[n_i]['coord']
        if is_edge_valid(v, u, shape_objs_l, r):
            edge_w = np.linalg.norm(u-v)
            plan_g.add_edge('dest', n_i, weight=edge_w)
    
    node_path = nx.shortest_path(plan_g, source='orig', target='dest', weight='weight', method='dijkstra')
    
    conf_path = [orig_conf]
    for n_i in node_path[1:]:
        n_i_coord = plan_g.nodes[n_i]['coord']
        conf_path.append(np.array([n_i_coord[0], n_i_coord[1], dest_conf[2]]))
    
    path_corridors = []
    for n_i_s, n_i_d in zip(node_path[:-1],node_path[1:]):
        v = plan_g.nodes[n_i_s]['coord']
        u = plan_g.nodes[n_i_d]['coord']
        path_corridors.append(corridor_v_u(v[0], v[1], u[0], u[1], r))
        
    return conf_path, path_corridors