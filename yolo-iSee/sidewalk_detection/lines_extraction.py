import cv2
import numpy as np
from sidewalk_detection.sidewalk_class import line

lower_canny = 150
upper_canny = 200
lower_gray = np.array([0, 0, 0])
upper_gray = np.array([200, 50, 255])



def preprocessing_gaussian(frame):
    frame = cv2.GaussianBlur(frame, (5,5), 0)
    return frame


def preprocessing_bilateral(frame):
    frame = cv2.bilateralFilter(frame, 5, 75, 75)
    return frame


def no_preprocessing(frame):
    return frame


def preprocessing_HSVcolor(frame):
    hsv = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_gray, upper_gray)

    frame = cv2.bitwise_and(frame, frame, mask= mask) 
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    return frame


def get_canny_bounds(frame, sigma=0.33):
    v = np.median(frame)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return lower, upper


def get_region(frame):
    height, width = frame.shape
    polygon = np.array([
                       [(int(0*width/1920), height), (int(520*width/1920), int(600*height/1080)), 
                       (int(1400*width/1920), int(600*height/1080)), (int(width+0*width/1920), height)]
                       ])
    mask = np.zeros_like(frame)
    mask = cv2.fillPoly(mask, polygon, 255)
    mask = cv2.bitwise_and(frame, mask)
    return mask


def conversion_to_lines(lines):
    res = []
    for li in lines:
        x1, y1, x2, y2 = li.reshape(4)
        l = line()
        l.points_from_coord(x1, y1, x2, y2)
        res.append(l)

    return res


#erase non significant lines (e.g. horizontal et simila)
def lines_scraping(lines: list):
    res=[]
    for line in lines:
        if line.slope is not None:
            if(line.slope > 0.3 or line.slope < -0.3):
                res.append(line)
        else:
            res.append(line)
    
    return res


def factory_lines(frame):
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = preprocessing_gaussian(frame)    
 
    lower_canny, upper_canny = get_canny_bounds(frame)
    frame = cv2.Canny(frame, lower_canny, upper_canny)
    frame = get_region(frame)

    lines = cv2.HoughLinesP(frame, 2, np.pi/180, threshold=220, minLineLength=130, maxLineGap=35)
    if(lines is None):
        return None

    lines = conversion_to_lines(lines)
    lines = lines_scraping(lines)
    return lines

