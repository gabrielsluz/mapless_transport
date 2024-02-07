"""
Made thinking of a (50, 50) world.
"""

obstacle_l_dict = {
    # EASY: a greedy agent can easily find a path to the goal.
    'empty': [],
    'circle_line': [
        {'name':'Circle', 'pos':(25.0, 0.0), 'radius':1.5},
        {'name':'Circle', 'pos':(25.0, 10.0), 'radius':1.5},
        {'name':'Circle', 'pos':(25.0, 20.0), 'radius':1.5},
        {'name':'Circle', 'pos':(25.0, 30.0), 'radius':1.5},
        {'name':'Circle', 'pos':(25.0, 40.0), 'radius':1.5},
        {'name':'Circle', 'pos':(25.0, 50.0), 'radius':1.5},
    ],
    'small_4_circles': [
        {'name':'Circle', 'pos':(17.0, 17.0), 'radius':2.0},
        {'name':'Circle', 'pos':(34.0, 17.0), 'radius':2.0},
        {'name':'Circle', 'pos':(17.0, 34.0), 'radius':2.0},
        {'name':'Circle', 'pos':(34.0, 34.0), 'radius':2.0},
    ],
    '1_circle': [
        {'name':'Circle', 'pos':(25.0, 25.0), 'radius':10.0},
    ],
    '1_rectangle': [
        {'name':'Rectangle', 'pos':(25.0, 25.0), 'height':22.0, 'width':12.0},
    ],
    '1_triangle': [
        {'name':'Polygon', 'pos':(25.0, 25.0), 'vertices':[(0,0), (22,0), (0,27)]},
    ],
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
    '25_circles': [
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
        # Upper 3 row:
        {'name':'Circle', 'pos':(12.5, 12.5), 'radius':2.0},
        {'name':'Circle', 'pos':(25.0, 12.5), 'radius':2.0},
        {'name':'Circle', 'pos':(37.5, 12.5), 'radius':2.0},
        # Middle 3 row:
        {'name':'Circle', 'pos':(12.5, 25.0), 'radius':2.0},
        {'name':'Circle', 'pos':(25.0, 25.0), 'radius':2.0},
        {'name':'Circle', 'pos':(37.5, 25.0), 'radius':2.0},
        # Lower 3 row:
        {'name':'Circle', 'pos':(12.5, 37.5), 'radius':2.0},
        {'name':'Circle', 'pos':(25.0, 37.5), 'radius':2.0},
        {'name':'Circle', 'pos':(37.5, 37.5), 'radius':2.0},
    ],  
    # Grid of 49 circles euqally spaced:
    '49_circles': [
        # Row 0:
        {'name':'Circle', 'pos':(0*8.33, 0), 'radius':2.0},
        {'name':'Circle', 'pos':(1*8.33, 0), 'radius':2.0},
        {'name':'Circle', 'pos':(2*8.33, 0), 'radius':2.0},
        {'name':'Circle', 'pos':(3*8.33, 0), 'radius':2.0},
        {'name':'Circle', 'pos':(4*8.33, 0), 'radius':2.0},
        {'name':'Circle', 'pos':(5*8.33, 0), 'radius':2.0},
        {'name':'Circle', 'pos':(6*8.33, 0), 'radius':2.0},
        # Row 1:
        {'name':'Circle', 'pos':(0*8.33, 8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(1*8.33, 8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(2*8.33, 8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(3*8.33, 8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(4*8.33, 8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(5*8.33, 8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(6*8.33, 8.33), 'radius':2.0},
        # Row 2:
        {'name':'Circle', 'pos':(0*8.33, 2*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(1*8.33, 2*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(2*8.33, 2*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(3*8.33, 2*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(4*8.33, 2*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(5*8.33, 2*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(6*8.33, 2*8.33), 'radius':2.0},
        # Row 3:
        {'name':'Circle', 'pos':(0*8.33, 3*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(1*8.33, 3*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(2*8.33, 3*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(3*8.33, 3*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(4*8.33, 3*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(5*8.33, 3*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(6*8.33, 3*8.33), 'radius':2.0},
        # Row 4:
        {'name':'Circle', 'pos':(0*8.33, 4*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(1*8.33, 4*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(2*8.33, 4*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(3*8.33, 4*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(4*8.33, 4*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(5*8.33, 4*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(6*8.33, 4*8.33), 'radius':2.0},
        # Row 5:
        {'name':'Circle', 'pos':(0*8.33, 5*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(1*8.33, 5*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(2*8.33, 5*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(3*8.33, 5*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(4*8.33, 5*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(5*8.33, 5*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(6*8.33, 5*8.33), 'radius':2.0},
        # Row 6:
        {'name':'Circle', 'pos':(0*8.33, 6*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(1*8.33, 6*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(2*8.33, 6*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(3*8.33, 6*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(4*8.33, 6*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(5*8.33, 6*8.33), 'radius':2.0},
        {'name':'Circle', 'pos':(6*8.33, 6*8.33), 'radius':2.0},
    ],
    '25_rectangles': [
        # Upper left:
        {'name': 'Rectangle', 'pos': (6.25, 6.25), 'height': 2.0, 'width': 5.0},
        {'name': 'Rectangle', 'pos': (18.75, 6.25), 'height': 2.0, 'width': 5.0},
        {'name': 'Rectangle', 'pos': (6.25, 18.75), 'height': 2.0, 'width': 5.0},
        {'name': 'Rectangle', 'pos': (18.75, 18.75), 'height': 2.0, 'width': 5.0},
        # Upper right:
        {'name': 'Rectangle', 'pos': (31.25, 6.25), 'height': 2.0, 'width': 5.0},
        {'name': 'Rectangle', 'pos': (43.75, 6.25), 'height': 2.0, 'width': 5.0},
        {'name': 'Rectangle', 'pos': (31.25, 18.75), 'height': 2.0, 'width': 5.0},
        {'name': 'Rectangle', 'pos': (43.75, 18.75), 'height': 2.0, 'width': 5.0},
        # Lower left:
        {'name': 'Rectangle', 'pos': (6.25, 31.25), 'height': 2.0, 'width': 5.0},
        {'name': 'Rectangle', 'pos': (18.75, 31.25), 'height': 2.0, 'width': 5.0},
        {'name': 'Rectangle', 'pos': (6.25, 43.75), 'height': 2.0, 'width': 5.0},
        {'name': 'Rectangle', 'pos': (18.75, 43.75), 'height': 2.0, 'width': 5.0},
        # Lower right:
        {'name': 'Rectangle', 'pos': (31.25, 31.25), 'height': 2.0, 'width': 5.0},
        {'name': 'Rectangle', 'pos': (43.75, 31.25), 'height': 2.0, 'width': 5.0},
        {'name': 'Rectangle', 'pos': (31.25, 43.75), 'height': 2.0, 'width': 5.0},
        {'name': 'Rectangle', 'pos': (43.75, 43.75), 'height': 2.0, 'width': 5.0},
        # Upper 3 row:
        {'name': 'Rectangle', 'pos': (12.5, 12.5), 'height': 2.0, 'width': 5.0},
        {'name': 'Rectangle', 'pos': (25.0, 12.5), 'height': 2.0, 'width': 5.0},
        {'name': 'Rectangle', 'pos': (37.5, 12.5), 'height': 2.0, 'width': 5.0},
        # Middle 3 row:
        {'name': 'Rectangle', 'pos': (12.5, 25.0), 'height': 2.0, 'width': 5.0},
        {'name': 'Rectangle', 'pos': (25.0, 25.0), 'height': 2.0, 'width': 5.0},
        {'name': 'Rectangle', 'pos': (37.5, 25.0), 'height': 2.0, 'width': 5.0},
        # Lower 3 row:
        {'name': 'Rectangle', 'pos': (12.5, 37.5), 'height': 2.0, 'width': 5.0},
        {'name': 'Rectangle', 'pos': (25.0, 37.5), 'height': 2.0, 'width': 5.0},
        {'name': 'Rectangle', 'pos': (37.5, 37.5), 'height': 2.0, 'width': 5.0},
    ],
    '25_small_triangles': [
        # Upper left:
        {'name': 'Polygon', 'pos': (6.25, 6.25), 'vertices': [(5, 8.25), (7, 8.25), (6, 6.25)]},
        {'name': 'Polygon', 'pos': (18.75, 6.25), 'vertices': [(17.75, 8.25), (19.75, 8.25), (18.75, 6.25)]},
        {'name': 'Polygon', 'pos': (6.25, 18.75), 'vertices': [(5, 20.75), (7, 20.75), (6, 18.75)]},
        {'name': 'Polygon', 'pos': (18.75, 18.75), 'vertices': [(17.75, 20.75), (19.75, 20.75), (18.75, 18.75)]},
        # Upper right:
        {'name': 'Polygon', 'pos': (31.25, 6.25), 'vertices': [(30.25, 8.25), (32.25, 8.25), (31.25, 6.25)]},
        {'name': 'Polygon', 'pos': (43.75, 6.25), 'vertices': [(42.75, 8.25), (44.75, 8.25), (43.75, 6.25)]},
        {'name': 'Polygon', 'pos': (31.25, 18.75), 'vertices': [(30.25, 20.75), (32.25, 20.75), (31.25, 18.75)]},
        {'name': 'Polygon', 'pos': (43.75, 18.75), 'vertices': [(42.75, 20.75), (44.75, 20.75), (43.75, 18.75)]},
        # Lower left:
        {'name': 'Polygon', 'pos': (6.25, 31.25), 'vertices': [(5, 33.25), (7, 33.25), (6, 31.25)]},
        {'name': 'Polygon', 'pos': (18.75, 31.25), 'vertices': [(17.75, 33.25), (19.75, 33.25), (18.75, 31.25)]},
        {'name': 'Polygon', 'pos': (6.25, 43.75), 'vertices': [(5, 45.75), (7, 45.75), (6, 43.75)]},
        {'name': 'Polygon', 'pos': (18.75, 43.75), 'vertices': [(17.75, 45.75), (19.75, 45.75), (18.75, 43.75)]},
        # Lower right:
        {'name': 'Polygon', 'pos': (31.25, 31.25), 'vertices': [(30.25, 33.25), (32.25, 33.25), (31.25, 31.25)]},
        {'name': 'Polygon', 'pos': (43.75, 31.25), 'vertices': [(42.75, 33.25), (44.75, 33.25), (43.75, 31.25)]},
        {'name': 'Polygon', 'pos': (31.25, 43.75), 'vertices': [(30.25, 45.75), (32.25, 45.75), (31.25, 43.75)]},
        {'name': 'Polygon', 'pos': (43.75, 43.75), 'vertices': [(42.75, 45.75), (44.75, 45.75), (43.75, 43.75)]},
        # Upper 3 row:
        {'name': 'Polygon', 'pos': (12.5, 12.5), 'vertices': [(11.5, 14.5), (13.5, 14.5), (12.5, 12.5)]},
        {'name': 'Polygon', 'pos': (25.0, 12.5), 'vertices': [(24.0, 14.5), (26.0, 14.5), (25.0, 12.5)]},
        {'name': 'Polygon', 'pos': (37.5, 12.5), 'vertices': [(36.5, 14.5), (38.5, 14.5), (37.5, 12.5)]},
        # Middle 3 row:
        {'name': 'Polygon', 'pos': (12.5, 25.0), 'vertices': [(11.5, 27.0), (13.5, 27.0), (12.5, 25.0)]},
        {'name': 'Polygon', 'pos': (25.0, 25.0), 'vertices': [(24.0, 27.0), (26.0, 27.0), (25.0, 25.0)]},
        {'name': 'Polygon', 'pos': (37.5, 25.0), 'vertices': [(36.5, 27.0), (38.5, 27.0), (37.5, 25.0)]},
        # Lower 3 row:
        {'name': 'Polygon', 'pos': (12.5, 37.5), 'vertices': [(11.5, 39.5), (13.5, 39.5), (12.5, 37.5)]},
        {'name': 'Polygon', 'pos': (25.0, 37.5), 'vertices': [(24.0, 39.5), (26.0, 39.5), (25.0, 37.5)]},
        {'name': 'Polygon', 'pos': (37.5, 37.5), 'vertices': [(36.5, 39.5), (38.5, 39.5), (37.5, 37.5)]},
    ],
    '25_triangles': [
        # Upper left:
        {'name': 'Polygon', 'pos': (6.25, 6.25), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        {'name': 'Polygon', 'pos': (18.75, 6.25), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        {'name': 'Polygon', 'pos': (6.25, 18.75), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        {'name': 'Polygon', 'pos': (18.75, 18.75), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        # Upper right:
        {'name': 'Polygon', 'pos': (31.25, 6.25), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        {'name': 'Polygon', 'pos': (43.75, 6.25), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        {'name': 'Polygon', 'pos': (31.25, 18.75), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        {'name': 'Polygon', 'pos': (43.75, 18.75), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        # Lower left:
        {'name': 'Polygon', 'pos': (6.25, 31.25), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        {'name': 'Polygon', 'pos': (18.75, 31.25), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        {'name': 'Polygon', 'pos': (6.25, 43.75), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        {'name': 'Polygon', 'pos': (18.75, 43.75), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        # Lower right:
        {'name': 'Polygon', 'pos': (31.25, 31.25), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        {'name': 'Polygon', 'pos': (43.75, 31.25), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        {'name': 'Polygon', 'pos': (31.25, 43.75), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        {'name': 'Polygon', 'pos': (43.75, 43.75), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        # Upper 3 row:
        {'name': 'Polygon', 'pos': (12.5, 12.5), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        {'name': 'Polygon', 'pos': (25.0, 12.5), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        {'name': 'Polygon', 'pos': (37.5, 12.5), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        # Middle 3 row:
        {'name': 'Polygon', 'pos': (12.5, 25.0), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        {'name': 'Polygon', 'pos': (25.0, 25.0), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        {'name': 'Polygon', 'pos': (37.5, 25.0), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        # Lower 3 row:
        {'name': 'Polygon', 'pos': (12.5, 37.5), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        {'name': 'Polygon', 'pos': (25.0, 37.5), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
        {'name': 'Polygon', 'pos': (37.5, 37.5), 'vertices': [(0, 0), (5, 0), (2.5, 5)]},
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
    'random_shapes_on_path': [
        # Upper left - Rectangles, Triangles, and Circles:
        {'name': 'Rectangle', 'pos': (2.0, 2.0), 'height': 3.0, 'width': 4.0},
        {'name': 'Rectangle', 'pos': (12.0, 7.0), 'height': 2.5, 'width': 5.5},
        {'name': 'Rectangle', 'pos': (22.0, 12.0), 'height': 2.0, 'width': 6.0},
        {'name': 'Rectangle', 'pos': (4.0, 17.0), 'height': 2.5, 'width': 4.5},
        {'name': 'Rectangle', 'pos': (15.0, 22.0), 'height': 3.0, 'width': 4.0},
        {'name': 'Rectangle', 'pos': (26.0, 27.0), 'height': 2.5, 'width': 5.0},
        
        {'name': 'Polygon', 'pos': (7.0, 22.0), 'vertices': [(6, 25), (8, 25), (7, 22)]},
        {'name': 'Polygon', 'pos': (17.0, 27.0), 'vertices': [(16, 30), (18, 30), (17, 27)]},
        {'name': 'Polygon', 'pos': (27.0, 32.0), 'vertices': [(26, 35), (28, 35), (27, 32)]},
        
        {'name': 'Circle', 'pos': (12.0, 12.0), 'radius': 2.5},
        {'name': 'Circle', 'pos': (22.0, 17.0), 'radius': 3.0},
        {'name': 'Circle', 'pos': (32.0, 22.0), 'radius': 3.5},
        {'name': 'Circle', 'pos': (42.0, 27.0), 'radius': 2.5},
    
        # Lower left - Rectangles, Triangles, and Circles:
        {'name': 'Rectangle', 'pos': (2.0, 32.0), 'height': 3.0, 'width': 4.0},
        {'name': 'Rectangle', 'pos': (12.0, 37.0), 'height': 2.5, 'width': 5.5},
        {'name': 'Rectangle', 'pos': (22.0, 42.0), 'height': 2.0, 'width': 6.0},
        {'name': 'Rectangle', 'pos': (4.0, 47.0), 'height': 2.5, 'width': 4.5},
        {'name': 'Rectangle', 'pos': (15.0, 52.0), 'height': 3.0, 'width': 4.0},
        {'name': 'Rectangle', 'pos': (26.0, 57.0), 'height': 2.5, 'width': 5.0},
        
        {'name': 'Polygon', 'pos': (7.0, 52.0), 'vertices': [(6, 55), (8, 55), (7, 52)]},
        {'name': 'Polygon', 'pos': (17.0, 57.0), 'vertices': [(16, 60), (18, 60), (17, 57)]},
        {'name': 'Polygon', 'pos': (27.0, 62.0), 'vertices': [(26, 65), (28, 65), (27, 62)]},
        
        {'name': 'Circle', 'pos': (12.0, 42.0), 'radius': 2.5},
        {'name': 'Circle', 'pos': (22.0, 47.0), 'radius': 3.0},
        {'name': 'Circle', 'pos': (32.0, 52.0), 'radius': 3.5},
        {'name': 'Circle', 'pos': (42.0, 57.0), 'radius': 2.5},
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

    # Pushing maps => thought for a circular object of radius 4
    # 1_circle, 1_rectangle, 1_triangle are also good.
    'frame': [
        {'name':'Rectangle', 'pos':(25.0, 0.0), 'height':5.0, 'width':50.0},
        {'name':'Rectangle', 'pos':(25.0, 50.0), 'height':5.0, 'width':50.0},
        {'name':'Rectangle', 'pos':(0.0, 25.0), 'height':50.0, 'width':5.0},
        {'name':'Rectangle', 'pos':(50.0, 25.0), 'height':50.0, 'width':5.0},
    ],

    'horizontal_corridor': [
        {'name':'Rectangle', 'pos':(25.0, 0.0), 'height':25.0, 'width':50.0},
        {'name':'Rectangle', 'pos':(25.0, 50.0), 'height':25.0, 'width':50.0},
    ],

    'vertical_corridor': [
        {'name':'Rectangle', 'pos':(0.0, 25.0), 'height':50.0, 'width':25.0},
        {'name':'Rectangle', 'pos':(50.0, 25.0), 'height':50.0, 'width':25.0},
    ],

    '4_circles_wide': [
        {'name':'Circle', 'pos':(12.0, 12.0), 'radius':4.0},
        {'name':'Circle', 'pos':(40.0, 12.0), 'radius':4.0},
        {'name':'Circle', 'pos':(12.0, 40.0), 'radius':4.0},
        {'name':'Circle', 'pos':(40.0, 40.0), 'radius':4.0},
    ],


    # Large pushing maps: 100x100
    'parallel_walls': [
        # Frame
        {'name':'Rectangle', 'pos':(50.0, 0.0), 'height':5.0, 'width':100.0},
        {'name':'Rectangle', 'pos':(50.0, 100.0), 'height':5.0, 'width':100.0},
        {'name':'Rectangle', 'pos':(0.0, 50.0), 'height':100.0, 'width':5.0},
        {'name':'Rectangle', 'pos':(100.0, 50.0), 'height':100.0, 'width':5.0},
        # Inside walls
        {'name':'Rectangle', 'pos':(30.0, 75.0), 'height':75.0, 'width':5.0},
        {'name':'Rectangle', 'pos':(60.0, 25.0), 'height':75.0, 'width':5.0},
    ],
    'big_U': [
        # Frame
        {'name':'Rectangle', 'pos':(50.0, 0.0), 'height':5.0, 'width':100.0},
        {'name':'Rectangle', 'pos':(50.0, 100.0), 'height':5.0, 'width':100.0},
        {'name':'Rectangle', 'pos':(0.0, 50.0), 'height':100.0, 'width':5.0},
        {'name':'Rectangle', 'pos':(100.0, 50.0), 'height':100.0, 'width':5.0},
        # Inside walls
        {'name':'Rectangle', 'pos':(33.0, 50.0), 'height':50.0, 'width':3.0},
        {'name':'Rectangle', 'pos':(66.0, 50.0), 'height':50.0, 'width':3.0},
        {'name':'Rectangle', 'pos':(49.5, 75.0), 'height':3.0, 'width':33.0},
        
    ],
    'big_sparse_2':[
        # Frame
        {'name':'Rectangle', 'pos':(50.0, 0.0), 'height':1.0, 'width':100.0},
        {'name':'Rectangle', 'pos':(50.0, 100.0), 'height':1.0, 'width':100.0},
        {'name':'Rectangle', 'pos':(0.0, 50.0), 'height':100.0, 'width':1.0},
        {'name':'Rectangle', 'pos':(100.0, 50.0), 'height':100.0, 'width':1.0},

        {'name':'Circle', 'pos':(50.0, 50.0), 'radius':12.0},
        {'name':'Rectangle', 'pos':(80.0, 10.0), 'height':12.0, 'width':12.0},
        {'name':'Circle', 'pos':(50.0, 10.0), 'radius':5.0},
        {'name':'Rectangle', 'pos':(10.0, 40.0), 'height':25.0, 'width':4.0},
        {'name': 'Polygon', 'pos':(28, 86.0), 'vertices': [(5,45), (14,40), (22,47)]},
        {'name': 'Polygon', 'pos':(80, 80.0), 'vertices': [(31, 46.0), (43,28), (47.5,46.3)]},
    ],

    # Very large pushing maps: 200x200
    'very_big_U': [
        # Frame
        {'name':'Rectangle', 'pos':(100.0, 0.0), 'height':5.0, 'width':200.0},
        {'name':'Rectangle', 'pos':(100.0, 200.0), 'height':5.0, 'width':200.0},
        {'name':'Rectangle', 'pos':(0.0, 100.0), 'height':200.0, 'width':5.0},
        {'name':'Rectangle', 'pos':(200.0, 100.0), 'height':200.0, 'width':5.0},
        # Inside walls
        {'name':'Rectangle', 'pos':(66.0, 100.0), 'height':100.0, 'width':3.0},
        {'name':'Rectangle', 'pos':(122.0, 100.0), 'height':100.0, 'width':3.0},
        {'name':'Rectangle', 'pos':(98.0, 150.0), 'height':3.0, 'width':66.0}, 
    ],
    'very_big_sparse_2':[
        # Frame
        {'name':'Rectangle', 'pos':(100.0, 0.0), 'height':5.0, 'width':200.0},
        {'name':'Rectangle', 'pos':(100.0, 200.0), 'height':5.0, 'width':200.0},
        {'name':'Rectangle', 'pos':(0.0, 100.0), 'height':200.0, 'width':5.0},
        {'name':'Rectangle', 'pos':(200.0, 100.0), 'height':200.0, 'width':5.0},

        {'name':'Circle', 'pos':(100.0, 100.0), 'radius':24.0},
        {'name':'Rectangle', 'pos':(160.0, 20.0), 'height':1.5*24.0, 'width':1.5*24.0},
        {'name':'Circle', 'pos':(100.0, 20.0), 'radius':1.5*10.0},
        {'name':'Rectangle', 'pos':(20.0, 80.0), 'height':1.5*50.0, 'width':1.5*8.0},
        {'name': 'Polygon', 'pos':(56, 172.0), 'vertices': [(2*10,2*90), (2*27,2*80), (2*44,2*94)]},
        {'name': 'Polygon', 'pos':(160, 160.0), 'vertices': [(4*31, 4*46.0), (4*43,4*28), (4*47.5,4*46.3)]},
    ],
    'very_big_parallel_walls': [
        # Frame
        {'name':'Rectangle', 'pos':(100.0, 0.0), 'height':5.0, 'width':200.0},
        {'name':'Rectangle', 'pos':(100.0, 200.0), 'height':5.0, 'width':200.0},
        {'name':'Rectangle', 'pos':(0.0, 100.0), 'height':200.0, 'width':5.0},
        {'name':'Rectangle', 'pos':(200.0, 100.0), 'height':200.0, 'width':5.0},
        # Inside walls
        {'name':'Rectangle', 'pos':(2*30.0, 2*75.0), 'height':2*75.0, 'width':5.0},
        {'name':'Rectangle', 'pos':(2*60.0, 2*25.0), 'height':2*75.0, 'width':5.0},
    ],

}