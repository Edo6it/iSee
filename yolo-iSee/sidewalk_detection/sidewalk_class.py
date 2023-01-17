import numpy as np
import cv2
import math


def dist(a, b):
    return np.linalg.norm(np.array(a)-np.array(b))


class line:
    def __init__(self):
        self.point_low = (0, 1080)
        self.point_up = (0, 0)
        self.coord = (0, 1080, 0, 0)
        self.slope = None
        self.y_init = None


    def points_from_line(self, line):
        if(self.__class__ != line.__class__):
            print("Error type")
            exit(0)

        self.coord = line.coord
        self.point_low = line.point_low
        self.point_up = line.point_up
        self.init_slope()


    def points_from_coord(self, x1: int, y1: int, x2: int, y2: int):
        pointA = (x1, y1)
        pointB = (x2, y2)

        if(pointA[1] < pointB[1]):
            tmp = pointA
            pointA = pointB
            pointB = tmp
        
        self.point_low = pointA
        self.point_up = pointB
        self.coord = (self.point_low[0], self.point_low[1], self.point_up[0], self.point_up[1])
        self.init_slope()


    def points_from_params(self, slope: float, y_init: int, line_lenght=3/5, img_dim=1080):
        #only for final lines, the averged ones 
        y1 = img_dim
        y2 = int(y1 * (line_lenght))
        x1 = int((y1 - y_init) // slope)
        x2 = int((y2 - y_init) // slope)

        self.point_low = (x1, y1)
        self.point_up = (x2, y2)
        self.coord = (x1, y1, x2, y2)
        self.slope = slope
        self.y_init = y_init


    def init_slope(self):
        if(self.point_low[0] == self.point_up[0]):
            #technically an infinite slope, practically we choose a very steep one to represent the verticality
            param = np.polyfit((self.point_low[0]+1, self.point_up[0]), (self.point_low[1], self.point_up[1]), 1)
        else:
            param = np.polyfit((self.point_low[0], self.point_up[0]), (self.point_low[1], self.point_up[1]), 1)
            
        self.slope, self.y_init = param[0], param[1]

    
    def dist_from_line(self, line):
        if(self.__class__ != line.__class__):
            print("Error type")
            exit(0)

        return dist(self.coord, line.coord)


    def mean_point(self):
        med_x = int((self.point_low[0] + self.point_up[0]) / 2)
        med_y = int((self.point_low[1] + self.point_up[1]) / 2)
        return (med_x, med_y)


    def add_line(self, frame, color=(0,0,0), size=10):
        frame = cv2.line(frame, self.point_low, self.point_up, color, size, cv2.LINE_AA)
        return frame





class sidewalk:
    def __init__(self):
        self.count = 0
        self.left_line = line()
        self.right_line = line()
        self.mid_line = line()
        self.width_up = [0]*20
        self.width_low = [0]*20
        self.mean_width_up = 0
        self.mean_width_low = 0
        self.median_color = [0, 0, 0]


    #update lines based on previous info
    def update(self, lineA: line, lineB: line, frame):

        if (lineA is None and lineB is None):
            left_line, right_line = self.update_no_lines()
        elif (lineB is None):
            left_line, right_line = self.update_one_line(lineA)
        else:
            left_line, right_line = self.update_two_lines(lineA, lineB)

        self.set_lines(left_line, right_line)

        self.set_mid_line()
        self.set_width(lineA, lineB)

        self.mean_width_low = get_mean_width(self.width_low)
        self.mean_width_up = get_mean_width(self.width_up)


    #if 10 consecutive frames have no lines, than sidewalk is no longer seen
    def update_no_lines(self):
        left_line = line()
        right_line = line()
        if(self.mean_width_low == 0):
            return left_line, right_line
        
        left_line.points_from_line(self.left_line)
        right_line.points_from_line(self.right_line)
        return left_line, right_line


    #get info from previous lines (if there are none, than get back to no lines)
    def update_one_line(self, lineA: line):
        left_line = line()
        right_line = line()
        
        if(self.mean_width_low == 0):
            return left_line, right_line

        d_r = dist(lineA.mean_point(), self.right_line.mean_point())
        d_l = dist(lineA.mean_point(), self.left_line.mean_point())

        if(d_r < d_l):
            right_line.points_from_line(lineA)
            point_up = (lineA.point_up[0] - self.mean_width_up, lineA.point_up[1])
            point_low = (lineA.point_low[0] - self.mean_width_low, lineA.point_low[1])
            left_line.points_from_coord(point_low[0], point_low[1], point_up[0], point_up[1])
        else:
            left_line.points_from_line(lineA)
            point_up = (lineA.point_up[0] + self.mean_width_up, lineA.point_up[1])
            point_low = (lineA.point_low[0] + self.mean_width_low, lineA.point_low[1])
            right_line.points_from_coord(point_low[0], point_low[1], point_up[0], point_up[1])
        
        return left_line, right_line


    #use the two lines as the new sidewalk boundary
    def update_two_lines(self, lineA: line, lineB: line):
        left_line = line()
        right_line = line()

        if(lineA.mean_point()[0] < lineB.mean_point()[0]):
            left_line.points_from_line(lineA)
            right_line.points_from_line(lineB)
        else:
            left_line.points_from_line(lineB)
            right_line.points_from_line(lineA)

        return left_line, right_line


    def set_lines(self, left_line: line, right_line: line):
        if(left_line.slope is None and right_line.slope is None):
            self.left_line = line()
            self.right_line = line()
            return 
        
        if(self.mean_width_low == 0):
            self.left_line = left_line
            self.right_line = right_line
            return
        
        x1, y1, x2, y2 = update_line_sidewalk(self.left_line, left_line, 0.8)
        self.left_line.points_from_coord(x1, y1, x2, y2)
        x1, y1, x2, y2 = update_line_sidewalk(self.right_line, right_line, 0.8)
        self.right_line.points_from_coord(x1, y1, x2, y2)

        return


    def set_width(self, lineA: line, lineB: line):
        self.width_low.pop(0)
        self.width_up.pop(0)

        if(lineA is None and lineB is None):
            self.width_up.append(0)
            self.width_low.append(0)
            return

        elif(lineB is None and self.mean_width_low == 0):
            self.width_up.append(0)
            self.width_low.append(0)
            return

        else:
            point_r = self.right_line.point_up
            point_l = self.left_line.point_up
            width_up = dist(point_l, point_r)
            self.width_up.append(width_up)

            point_r = self.right_line.point_low
            point_l = self.left_line.point_low
            width_low = dist(point_l, point_r)
            self.width_low.append(width_low)

        return


    def set_mid_line(self):
        if(self.right_line.slope is None and self.left_line.slope is None):
            self.mid_line = line()
            return
        
        #mean line of the two "edges" of the sidewalk
        x1, y1 = self.right_line.point_low[0]+self.left_line.point_low[0],   self.right_line.point_low[1]+self.left_line.point_low[1]
        x2, y2 = self.right_line.point_up[0]+self.left_line.point_up[0],   self.right_line.point_up[1]+self.left_line.point_up[1]
        self.mid_line.points_from_coord(int(x1/2), int(y1/2), int(x2/2), int(y2/2))


    def set_color(self, frame):
        if(self.right_line.slope is None and self.left_line.slope is None):
            self.median_color= [0, 0, 0]
            return

        polygon = np.array([
                       [self.left_line.point_low, self.left_line.point_up, self.right_line.point_up, self.right_line.point_low]
                       ])
        
        mask = np.zeros_like(frame)
        mask = cv2.fillPoly(mask, polygon, (255,255,255))

        colors = cv2.bitwise_and(frame, mask)
        colors = cv2.resize(colors, (192, 108), interpolation=cv2.INTER_AREA)

        pixels = []
        for row in colors:
            for pixel in row:
                if(pixel.sum() != 0):
                    pixels.append(pixel)
        pixels = np.array(pixels)

        self.median_color = np.median(pixels, axis=0)
        return





def update_line_sidewalk(self: line, line: line, sigma: float):
        
    x1 = int((1-sigma)*self.point_low[0] + sigma*line.point_low[0])
    y1 = int((1-sigma)*self.point_low[1] + sigma*line.point_low[1])

    x2 = int((1-sigma)*self.point_up[0] + sigma*line.point_up[0])
    y2 = int((1-sigma)*self.point_up[1] + sigma*line.point_up[1])

    return x1, y1, x2, y2


def get_mean_width(widths: list):
    wid = np.array(widths)

    if(np.size(wid) == 0):
        return 0

    wid = wid[wid!=0]

    if(np.size(wid) == 0):
        return 0
    else:
        return int(np.mean(wid))