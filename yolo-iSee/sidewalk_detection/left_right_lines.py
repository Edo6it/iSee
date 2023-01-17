import numpy as np
from sidewalk_detection.group_by_slopes import get_groups
from sidewalk_detection.sidewalk_class import line


def get_parameters(group: list):
    params = []
    for line in group:
        params.append((line.slope, line.y_init))

    return params


def avg_line(params: tuple, y_dimension=1080, line_lenght=3/4):
    if params.__len__() == 0:
        return None
    
    avg_param = np.average(params, axis=0)
    avg_line = line()
    avg_line.points_from_params(avg_param[0], avg_param[1], line_lenght, y_dimension)
    return avg_line


def left_right_lines(lines, frame, sidewalk_width):
    if(lines is None):
        return None, None
        
    #divide lines into groups using an algorithm (e.g. Kmeans)
    groupA, groupB = get_groups(lines)

    #get the parameters (slope etc) of each group and compute the left and right lines of the sidewalk
    paramsA = get_parameters(groupA)
    paramsB = get_parameters(groupB)
    lineA = avg_line(paramsA, frame.shape[0])
    lineB = avg_line(paramsB, frame.shape[0])

    if(lineA is None and lineB is not None):
        return lineB, lineA
    return lineA, lineB