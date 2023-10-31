"""
Módulo Python com funções para criar e usar mapas.
"""

import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry.polygon import orient

from research_envs.path_planning.descartes_mod.patch import PolygonPatch

def plot_map(obs_set, x_rng, y_rng):
    """
    Mostra o mapa de polígonos Shapely
    """
    fig = plt.figure(figsize=(8,5), dpi=100)
    ax = fig.add_subplot(111, aspect='equal') 

    for obs in obs_set:
        ax.add_patch(PolygonPatch(obs, facecolor='gray'))
        
    ax.set_xlim(x_rng[0], x_rng[1])
    ax.set_ylim(y_rng[0], y_rng[1])


def plot_map_path(obs_set, path_corridors, x_rng, y_rng):
    """
    Mostra o mapa com os caminhos em verde
    """
    fig = plt.figure(figsize=(8,5), dpi=100)
    ax = fig.add_subplot(111, aspect='equal') 

    for obs in obs_set:
        ax.add_patch(PolygonPatch(obs, facecolor='gray'))
    for corr in path_corridors:
        ax.add_patch(PolygonPatch(corr, facecolor='green'))
        
    ax.set_xlim(x_rng[0], x_rng[1])
    ax.set_ylim(y_rng[0], y_rng[1])
    plt.show()

def clean_next_dup(l):
    new_l = []
    last_e = None
    for e in l:
        if last_e is None or e != last_e:
            last_e = e
            new_l.append(e)
    return new_l

def pre_process_shapely(shape_objs_l):
    """
    Pre-processa o shape_objs_l para estar sem final repetido e 
    em ordem anti-horário
    """
    new_shape_objs_l = []
    for obj in shape_objs_l:
        coords = list(obj.exterior.coords)
        coords = clean_next_dup(coords)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        new_shape_objs_l.append(orient(Polygon(coords), sign=1.0))
    return new_shape_objs_l


def shapely_to_repr(shape_objs_l):
    shape_objs_l = pre_process_shapely(shape_objs_l)
    poly_coord_l = []
    for objs in shape_objs_l:
        poly_coord_l.append(list(objs.exterior.coords)[:-1])
    return poly_coord_l

def transform_to_range(shape_objs_l, x_rng, y_rng):
    array_l = []
    for obj in shape_objs_l:
        array_l.append(np.array(obj.exterior.coords))
    all_vs = np.concatenate(array_l).T
    i_x_rng = all_vs[0].min(), all_vs[0].max()
    i_y_rng = all_vs[1].min(), all_vs[1].max()

    scale_x = (x_rng[1] - x_rng[0]) / (i_x_rng[1] - i_x_rng[0])
    scale_y = (y_rng[1] - y_rng[0]) / (i_y_rng[1] - i_y_rng[0])
    off_x = x_rng[0] - scale_x*i_x_rng[0]
    off_y = y_rng[0] - scale_y*i_y_rng[0]
    new_shape_objs_l = []
    for obj in shape_objs_l:
        coords = list(obj.exterior.coords)
        trans_coords = [(off_x + scale_x*c[0], off_y + scale_y*c[1]) for c in coords]
        new_shape_objs_l.append(Polygon(trans_coords))
    return new_shape_objs_l