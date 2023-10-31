"""
Módulo Python que contém funções usados para o planjemento de caminhos
no mapa
"""
import numpy as np
import copy

from shapely.geometry import Polygon
from shapely.geometry import Point

# Funções geométricas no mapa

def corridor_v_u(vx, vy, ux, uy, r):
    """
    Cria um retângulo entre v e u, de largura 2r
    Pega a união com circulos centrados em v e u, de raio r.
    """
    # Ponto de singularidade
    if abs(vx - ux) < 1e-10:
        ux += 1e-12
    # Encontra a equação da reta entre v e u
    a = (vy - uy) / (vx - ux)
    b = uy - a*ux
    # Encontra o angulo entre a reta e o eixo x
    ang = np.arctan2(a, 1)
    # Encontra o vetor ortogonal com a reta, de tamanho r
    orth_x = -r*np.sin(ang)
    orth_y = r*np.cos(ang)
    # Cria os vértices
    rect_vs = np.array([
        [vx - orth_x, vy - orth_y],
        [ux - orth_x, uy - orth_y],
        [ux + orth_x, uy + orth_y],
        [vx + orth_x, vy + orth_y]
    ])
    rect_corr = Polygon(rect_vs)
    v_cir = Point(vx,vy).buffer(r)
    u_cir = Point(ux,uy).buffer(r)
    return v_cir.union(rect_corr).union(u_cir)


def is_edge_valid(v, u, shape_objs_l, r):
    """
    Verifica se a aresta é válida
    Entrada:
        v e u => np.arrays (x,y)
        shape_objs_l => lista de polígonos shapely, que representam os obstáculos sem Minkowski
        r => raio do círculo que representa o robô
    """
    corridor = corridor_v_u(v[0], v[1], u[0], u[1], r)
    for obj in shape_objs_l:
        if corridor.intersection(obj):
            return False
    return True

def smooth_conf_path(conf_path, shape_objs_l, r):
    """
    Tenta suavizar o caminho.
    """
    new_conf_path = []
    i = 0
    while i < len(conf_path):
        new_conf_path.append(copy.deepcopy(conf_path[i]))
        next_ind = i+1
        for j in range(len(conf_path)-1, i, -1):
            if is_edge_valid(conf_path[i][:2], conf_path[j][:2], shape_objs_l, r):
                next_ind = j
                break
        i = next_ind
    return new_conf_path  

def calc_path_cost(conf_path):
    """
    Soma a distância de cada passo do caminho.
    """
    sum_len = 0
    for c1, c2 in zip(conf_path[:-1],conf_path[1:]):
        sum_len += np.linalg.norm(c2[:2] - c1[:2])
    return sum_len