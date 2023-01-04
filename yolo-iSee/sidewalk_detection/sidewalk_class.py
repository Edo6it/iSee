import numpy as np
import cv2
from math import dist



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
        self.left_line = line()
        self.right_line = line()
        self.mid_line = line()
        self.width_up = 0
        self.width_low = 0


    def update(self, lineA: line, lineB: line, frame, sigma=1):
        if(self.right_line.slope is not None):
            return 

        if(lineA.mean_point()[0] < lineB.mean_point()[0]):
            self.left_line.points_from_line(lineA)
            self.right_line.points_from_line(lineB)
        else:
            self.left_line.points_from_line(lineB)
            self.right_line.points_from_line(lineA)

        #self.right_line = self.right_line * (1-sigma) + right * sigma
        #self.left_line = self.left_line * (1-sigma) + left * sigma

        self.set_width()
        self.set_mid_line()
        self.set_color(frame)


    def set_width(self):
        point_r = self.right_line.point_up
        point_l = self.left_line.point_up
        self.width_up = dist(point_l, point_r)

        point_r = self.right_line.point_low
        point_l = self.left_line.point_low
        self.width_low = dist(point_l, point_r)
        return


    def set_mid_line(self):
        #mean line of the two "edges" of the sidewalk
        x1, y1 = self.right_line.point_low[0]+self.left_line.point_low[0],   self.right_line.point_low[1]+self.left_line.point_low[1]
        x2, y2 = self.right_line.point_up[0]+self.left_line.point_up[0],   self.right_line.point_up[1]+self.left_line.point_up[1]
        self.mid_line.points_from_coord(int(x1/2), int(y1/2), int(x2/2), int(y2/2))


    def set_color(self, frame):
        polygon = np.array([
                       [self.left_line.point_low, self.left_line.point_up, self.right_line.point_up, self.right_line.point_low]
                       ])
        
        mask = np.zeros_like(frame)
        mask = cv2.fillPoly(mask, polygon, (255,255,255))
        #cv2.imwrite("C:\\Users\\ilcai\\Pictures\\side.png", mask)
        colors = cv2.bitwise_and(frame, mask)
        return


    def add_sidewalk(self, frame, color_left=(0,0,0), color_right=(255,255,255), size=7):
        frame = cv2.line(frame, self.left_line.point_low, self.left_line.point_up, color_left, size, cv2.LINE_AA)
        frame = cv2.line(frame, self.right_line.point_low, self.right_line.point_up, color_right, size, cv2.LINE_AA)
        frame = cv2.line(frame, self.mid_line.point_low, self.mid_line.point_up, (120,60,90), size, cv2.LINE_AA)
        return frame


    def information(self):
        #get the slope of the line, to understand if the sidewalk is straight or turning (left or right)
        if(self.mid_line.slope == 0):
            print("Errore!")
        elif(self.mid_line.slope > -0.2 and self.mid_line.slope < 0):
            print("Il marciapiede curva verso destra")
        elif(self.mid_line.slope > 0 and self.mid_line.slope < 0.2):
            print("Il marciapiede curva verso sinistra")
        else:
            print("Il marciapiede prosegue diritto")
        
        #get position of the sidewalk relatively to the person perspective
        point = self.mid_line.mean_point()
        reference_point = (960, 540)
        if(point[0] < reference_point[0]):
            print("sei sulla destra del marciapiede, accentrati verso sinistra")
        if(point[0] > reference_point[0]):
            print("sei sulla sinistra del marciapiede, accentrati verso destra")

