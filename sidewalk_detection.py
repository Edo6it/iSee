from xmlrpc.client import MAXINT
import cv2
import os
import numpy as np
from math import dist
os.environ['KMP_DUPLICATE_LIB_OK']='True'

img_path = "C:\\Users\\ilcai\\Pictures\\DatasetTest\\GoProModified\\VideoGoPro1_corrected\\GoPro1_2_Corrected.png"
lower_canny = 120
upper_canny = 180
dist_thresh = 400
stddev_thresh = 200


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


    def update(self, lineA: line, lineB: line, sigma=1):
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
        self.set_mid_line()


    def set_mid_line(self):
        #mean line of the two "edges" of the sidewalk
        x1, y1 = self.right_line.point_low[0]+self.left_line.point_low[0],   self.right_line.point_low[1]+self.left_line.point_low[1]
        x2, y2 = self.right_line.point_up[0]+self.left_line.point_up[0],   self.right_line.point_up[1]+self.left_line.point_up[1]
        self.mid_line.points_from_coord(int(x1/2), int(y1/2), int(x2/2), int(y2/2))


    def add_sidewalk(self, frame, color_left=(0,0,0), color_right=(255,255,255), size=7):
        frame = cv2.line(frame, self.left_line.point_low, self.left_line.point_up, color_left, size, cv2.LINE_AA)
        frame = cv2.line(frame, self.right_line.point_low, self.right_line.point_up, color_right, size, cv2.LINE_AA)
        frame = cv2.line(frame, self.mid_line.point_low, self.mid_line.point_up, (120,60,90), size, cv2.LINE_AA)
        return frame





def factory_lines(lines):
    if(lines is None):
        return None

    res = []
    for li in lines:
        x1, y1, x2, y2 = li.reshape(4)
        l = line()
        l.points_from_coord(x1, y1, x2, y2)
        res.append(l)

    return res


def lines_scraping(lines: list):
    res=[]
    for line in lines:
        if line.slope is not None:
            print(line.coord, "-->", line.slope)
            if(line.slope > 0.1 or line.slope < -0.1):
                res.append(line)
        else:
            print(line.coord, "-->", "inf")
            res.append(line)
    
    return res


def get_parameters(group: list):
    params = []
    for line in group:
        params.append((line.slope, line.y_init))

    return params


def avg_line(params: tuple, y_dimension=1080, line_lenght=3/4):
    avg_param = np.average(params, axis=0)
    avg_line = line()
    avg_line.points_from_params(avg_param[0], avg_param[1], line_lenght, y_dimension)

    return avg_line


def get_region(frame):
    height, width = frame.shape
    polygon = np.array([
                       [(-300, height), (520, 600), (1400, 600), (width+300, height)]
                       ])
    mask = np.zeros_like(frame)
    mask = cv2.fillPoly(mask, polygon, 255)
    cv2.imwrite("C:\\Users\\ilcai\\Pictures\\mask.png", mask)
    mask = cv2.bitwise_and(frame, mask)
    return mask


def get_canny_bounds(frame, sigma=0.33):
    v = np.median(frame)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return lower, upper


def get_groups(lines: list):
    #caso particolare, non ho linee, o ne ho solo una
    if len(lines) <= 1:
        return lines, []
    
    #caso particolare, ho due linee (se sono vicine, un solo gruppo, altrimenti due)
    #threshold value messo a 400 (per img a 1920*1080)
    if len(lines) == 2:
        distance = lines[0].dist_from_line(lines[1])
        if(distance > dist_thresh):
            return [lines[0]], [lines[1]]
        else:
            return lines, []
    
    #caso generico, linee>2
    groupA = []
    groupB = []

    #raggruppa le linee, sinistra e destra (non sai quale ma non importa)
    #calcolo punti medi
    med_points = []
    for line in lines:
        med_points.append(line.mean_point())
    
    #trovo i due punti medi più distanti tra loro
    A, B = 0, 0
    max= 0
    vec_dist = []
    for i in range(0, len(med_points)): 
        for j in range(i+1, len(med_points)):
            distance = dist(med_points[i], med_points[j])
            vec_dist.append(distance)
            if(distance > max):
                max = distance
                A, B = med_points[i], med_points[j]

    stddev = np.std(vec_dist)

    #se la std deviation è bassa, un solo gruppo (altrimenti 2)
    if(stddev < stddev_thresh):
        return lines, []
    
    #inserisco tutti i punti medi (linee) nei gruppi, in base a quale dei due punti precedenti è più vicino
    i = 0
    groupA = []
    groupB = []
    for med in med_points:
        if dist(A, med) < dist(B, med):
            groupA.append(lines[i])
        else:
            groupB.append(lines[i])
        i+=1
    
    return groupA, groupB


def left_right_lines(lines):
    groupA, groupB = get_groups(lines)

    paramsA = get_parameters(groupA)
    paramsB = get_parameters(groupB)

    lineA = avg_line(paramsA, frame.shape[0])
    lineB = avg_line(paramsB, frame.shape[0])

    return lineA, lineB


def get_lines(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (7,7), 0)

    lower_canny, upper_canny = get_canny_bounds(frame)
    frame = cv2.Canny(frame, lower_canny, upper_canny)
    frame = get_region(frame)

    cv2.imshow("canny_region", cv2.resize(frame, (960, 540), interpolation = cv2.INTER_AREA))
    cv2.waitKey(0)

    lines = cv2.HoughLinesP(frame, 2, np.pi/180, threshold=220, minLineLength=150, maxLineGap=35)
    
    if(lines is None):
        return None, None
    
    lines = factory_lines(lines)

    #togli linee non significative e potenzialmente sbagliate (es: orizzontali o quasi)
    lines = lines_scraping(lines)

    lineA, lineB = left_right_lines(lines)
    
    return lineA, lineB


def info_extract(frame, lineA: line, lineB: line, side: sidewalk):
    if(lineA is None and lineB is None):
        print("impossibile identificare marciapiede")
        return
    
    if(lineB is None):
        print("solo una linea trovata!")
        return

    print("entrambe le linee trovate!")

    #update of sidewalk model
    side.update(lineA, lineB)

    #get the slope of the line, to understand if the sidewalk is straight or turning (left or right)
    slope = side.mid_line.slope
    if(slope == 0):
        print("Errore!")
    elif(slope > -0.2 and slope < 0):
        print("Il marciapiede curva verso destra")
    elif(slope > 0 and slope < 0.2):
        print("Il marciapiede curva verso sinistra")
    else:
        print("Il marciapiede prosegue diritto")

    #get position of the sidewalk relatively to the person perspective
    med_x, med_y = side.mid_line.mean_point()
    point = (med_x, 540)
    reference_point = (960, 540)
    if(point[0] < reference_point[0]):
        print("sei sulla destra del marciapiede, accentrati verso sinistra")
    if(point[0] > reference_point[0]):
        print("sei sulla sinistra del marciapiede, accentrati verso destra")
    
    #show the line on screen
    frame = side.add_sidewalk(frame)
    cv2.imshow("Hough_lines", cv2.resize(frame, (960, 540), interpolation = cv2.INTER_AREA))
    cv2.waitKey(0)

    return side
    

def find(frames):
    side = sidewalk()

    for frame in frames:
        lineA, lineB = get_lines(frame)
        side = info_extract(frame, lineA, lineB, side)
            


frames = []
frame = cv2.imread(img_path)
frames.append(frame)
find(frames)