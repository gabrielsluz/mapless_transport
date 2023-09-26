"""
Made thinking of a (50, 50) world.
"""

obstacle_l_dict = {
    # EASY: a greedy agent can easily find a path to the goal.
    '4_circles': [
        {'name':'Circle', 'pos':(17.0, 17.0), 'radius':4.0},
        {'name':'Circle', 'pos':(34.0, 17.0), 'radius':4.0},
        {'name':'Circle', 'pos':(17.0, 34.0), 'radius':4.0},
        {'name':'Circle', 'pos':(34.0, 34.0), 'radius':4.0},
    ],
    'corridor': [
        {'name':'Rectangle', 'pos':(35.0, 25.0), 'height':40.0, 'width':3.0},
        {'name':'Rectangle', 'pos':(10.0, 25.0), 'height':40.0, 'width':3.0},
    ],
    'crooked_corridor': [
        # Right side:
        {'name':'Rectangle', 'pos':(35.0, 25.0), 'height':40.0, 'width':3.0},
        {'name':'Rectangle', 'pos':(28.0, 45.0), 'height':3.0, 'width':8.0},
        # Left side:
        {'name':'Rectangle', 'pos':(10.0, 10.0), 'height':10.0, 'width':3.0},
        {'name':'Rectangle', 'pos':(12.0, 16.5), 'height':3.0, 'width':10.0},
        {'name':'Rectangle', 'pos':(15.5, 23.5), 'height':10.0, 'width':3.0},
        {'name':'Rectangle', 'pos':(20.0, 30.0), 'height':3.0, 'width':10.0},
        {'name':'Rectangle', 'pos':(23.0, 35.0), 'height':5.0, 'width':3.0},
    ],
    # MEDIUM: a greedy agent with short memory can find a path to the goal.
    '16_circles': [
        # Upper left:
        {'name':'Circle', 'pos':(6.25, 6.25), 'radius':2.0},
        {'name':'Circle', 'pos':(18.75, 6.25), 'radius':2.0},
        {'name':'Circle', 'pos':(6.25, 18.75), 'radius':2.0},
        {'name':'Circle', 'pos':(18.75, 18.75), 'radius':2.0},
        # Upper right:
        {'name':'Circle', 'pos':(31.25, 6.25), 'radius':2.0},
        {'name':'Circle', 'pos':(43.75, 6.25), 'radius':2.0},
        {'name':'Circle', 'pos':(31.25, 18.75), 'radius':2.0},
        {'name':'Circle', 'pos':(43.75, 18.75), 'radius':2.0},
        # Lower left:
        {'name':'Circle', 'pos':(6.25, 31.25), 'radius':2.0},
        {'name':'Circle', 'pos':(18.75, 31.25), 'radius':2.0},
        {'name':'Circle', 'pos':(6.25, 43.75), 'radius':2.0},
        {'name':'Circle', 'pos':(18.75, 43.75), 'radius':2.0},
        # Lower right:
        {'name':'Circle', 'pos':(31.25, 31.25), 'radius':2.0},
        {'name':'Circle', 'pos':(43.75, 31.25), 'radius':2.0},
        {'name':'Circle', 'pos':(31.25, 43.75), 'radius':2.0},
        {'name':'Circle', 'pos':(43.75, 43.75), 'radius':2.0},
    ],
    'sparse_1':[
        {'name':'Circle', 'pos':(5.0, 5.0), 'radius':2.0},
        {'name':'Circle', 'pos':(10.0, 10.0), 'radius':5.0},
        {'name':'Circle', 'pos':(35.0, 35.0), 'radius':2.0},
        {'name':'Circle', 'pos':(45.0, 35.0), 'radius':2.0},
        {'name':'Circle', 'pos':(5.0, 35.0), 'radius':4.0},
        {'name':'Rectangle', 'pos':(25.0, 25.0), 'height':10.0, 'width':2.0}
    ],
    'sparse_2':[
        {'name':'Circle', 'pos':(25.0, 25.0), 'radius':8.0},
        {'name':'Rectangle', 'pos':(40.0, 5.0), 'height':9.0, 'width':9.0},
        {'name':'Circle', 'pos':(25.0, 5.0), 'radius':5.0},
        {'name':'Rectangle', 'pos':(5.0, 20.0), 'height':20.0, 'width':4.0},
        {'name': 'Polygon', 'pos':(14, 43.0), 'vertices': [(5,45), (14,40), (22,47)]},
        {'name': 'Polygon', 'pos':(40, 40.0), 'vertices': [(31, 46.0), (43,28), (47.5,46.3)]},
    ],
    'sparse_3':[
        {'name': 'Polygon', 'pos':(25, 25), 'vertices': [(4, 15), (14,3), (10,-13), (-4,-10), (-8, 3)]},
        {'name':'Circle', 'pos':(25.0, 5.0), 'radius':4.0},
        {'name':'Circle', 'pos':(5.0, 20.0), 'radius':5.0},
        {'name':'Circle', 'pos':(45.0, 20.0), 'radius':5.0},
        {'name':'Circle', 'pos':(5.0, 40.0), 'radius':5.0},
        {'name':'Circle', 'pos':(45.0, 40.0), 'radius':5.0},
        {'name':'Rectangle', 'pos':(5.0, 5.0), 'height':5.0, 'width':5.0},
        {'name':'Rectangle', 'pos':(45.0, 5.0), 'height':5.0, 'width':5.0},
        {'name':'Rectangle', 'pos':(5.0, 45.0), 'height':5.0, 'width':5.0},
        {'name':'Rectangle', 'pos':(45.0, 45.0), 'height':5.0, 'width':5.0},
    ],
    'small_U': [
        {'name':'Rectangle', 'pos':(17.5+12.5, 12.5+12.5), 'height':20.0, 'width':1.5},
        {'name':'Rectangle', 'pos':(5.0+12.5, 12.5+12.5), 'height':20.0, 'width':1.5},
        {'name':'Rectangle', 'pos':(11.25+12.5, 22.5+12.5), 'height':1.5, 'width':9.0},
        # Walls:
        {'name':'Rectangle', 'pos':(25.0, 0.0), 'height':20.0, 'width':50.0},
        {'name':'Rectangle', 'pos':(0.0, 25.0), 'height':50.0, 'width':20.0},
        {'name':'Rectangle', 'pos':(50, 25.0), 'height':50.0, 'width':20.0},
        {'name':'Rectangle', 'pos':(25.0, 50.0), 'height':20.0, 'width':50.0},
    ],
    'small_G': [
        {'name':'Rectangle', 'pos':(17.5+12.5, 17.5+12.5), 'height':10.0, 'width':1.5},
        {'name':'Rectangle', 'pos':(5.0+12.5, 12.5+12.5), 'height':20.0, 'width':1.5},
        {'name':'Rectangle', 'pos':(11.25+12.5, 22.5+12.5), 'height':1.5, 'width':9.0},
        {'name':'Rectangle', 'pos':(11.25+12.5, 2.5+12.5), 'height':1.5, 'width':9.0},
        {'name':'Rectangle', 'pos':(14.0+12.5, 13.5+12.5), 'height':1.5, 'width':4.0},
        # Walls:
        {'name':'Rectangle', 'pos':(25.0, 0.0), 'height':20.0, 'width':50.0},
        {'name':'Rectangle', 'pos':(0.0, 25.0), 'height':50.0, 'width':20.0},
        {'name':'Rectangle', 'pos':(50, 25.0), 'height':50.0, 'width':20.0},
        {'name':'Rectangle', 'pos':(25.0, 50.0), 'height':20.0, 'width':50.0},
    ],

    # Hard: a greedy agent may get stuck.
    'U': [
        {'name':'Rectangle', 'pos':(35.0, 25.0), 'height':40.0, 'width':3.0},
        {'name':'Rectangle', 'pos':(10.0, 25.0), 'height':40.0, 'width':3.0},
        {'name':'Rectangle', 'pos':(22.5, 45.0), 'height':3.0, 'width':18.0},
    ],
    'G': [
        {'name':'Rectangle', 'pos':(35.0, 35.0), 'height':20.0, 'width':3.0},
        {'name':'Rectangle', 'pos':(10.0, 25.0), 'height':40.0, 'width':3.0},
        {'name':'Rectangle', 'pos':(22.5, 45.0), 'height':3.0, 'width':18.0},
        {'name':'Rectangle', 'pos':(22.5, 5.0), 'height':3.0, 'width':18.0},
        {'name':'Rectangle', 'pos':(28.0, 27.0), 'height':3.0, 'width':8.0},
    ],
}


# - 4 obstáculos circulares
#         - Corredor
#         - U
#         - G
#         - Aleatório 1
#         - Aleatório 2
#         - Aleatório 3
#         - Complexo 1
#         - COmplexo 2