obstacle_l_dict = {
    'circle_middle_120x120': [
        {'name':'Rectangle', 'pos':(60.0, 0.0), 'height':1.0, 'width':120.0},
        {'name':'Rectangle', 'pos':(60.0, 120.0), 'height':1.0, 'width':120.0},
        {'name':'Rectangle', 'pos':(0.0, 60.0), 'height':120.0, 'width':1.0},
        {'name':'Rectangle', 'pos':(120.0, 60.0), 'height':120.0, 'width':1.0},

        {'name':'Circle', 'pos':(60.0, 60.0), 'radius':30.0},
    ],
    'parallel_walls_120x120': [
        {'name':'Rectangle', 'pos':(60.0, 0.0), 'height':1.0, 'width':120.0},
        {'name':'Rectangle', 'pos':(60.0, 120.0), 'height':1.0, 'width':120.0},
        {'name':'Rectangle', 'pos':(0.0, 60.0), 'height':120.0, 'width':1.0},
        {'name':'Rectangle', 'pos':(120.0, 60.0), 'height':120.0, 'width':1.0},

        # Inside walls
        {'name':'Rectangle', 'pos':(40.0, 80.0), 'height':80.0, 'width':3.0},
        {'name':'Rectangle', 'pos':(80.0, 40.0), 'height':80.0, 'width':3.0},
    ],
    'four_circles_120x120': [
        {'name':'Rectangle', 'pos':(60.0, 0.0), 'height':1.0, 'width':120.0},
        {'name':'Rectangle', 'pos':(60.0, 120.0), 'height':1.0, 'width':120.0},
        {'name':'Rectangle', 'pos':(0.0, 60.0), 'height':120.0, 'width':1.0},
        {'name':'Rectangle', 'pos':(120.0, 60.0), 'height':120.0, 'width':1.0},

        {'name':'Circle', 'pos':(35.0, 35.0), 'radius':10.0},
        {'name':'Circle', 'pos':(35.0, 85.0), 'radius':10.0},
        {'name':'Circle', 'pos':(85.0, 35.0), 'radius':10.0},
        {'name':'Circle', 'pos':(85.0, 85.0), 'radius':10.0},
    ],
}