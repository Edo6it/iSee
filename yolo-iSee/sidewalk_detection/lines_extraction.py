import cv2
import numpy as np
from sidewalk_class import line
lower_canny = 120
upper_canny = 180



def get_canny_bounds(frame, sigma=0.33):
    v = np.median(frame)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return lower, upper


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


def conversion_to_lines(lines):
    res = []
    for li in lines:
        x1, y1, x2, y2 = li.reshape(4)
        l = line()
        l.points_from_coord(x1, y1, x2, y2)
        res.append(l)

    return res


#togli linee non significative e potenzialmente sbagliate (es: orizzontali o quasi)
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


def factory_lines(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (7,7), 0)

    lower_canny, upper_canny = get_canny_bounds(frame)
    frame = cv2.Canny(frame, lower_canny, upper_canny)
    frame = get_region(frame)

    cv2.imshow("canny_region", cv2.resize(frame, (960, 540), interpolation = cv2.INTER_AREA))
    cv2.waitKey(0)

    lines = cv2.HoughLinesP(frame, 2, np.pi/180, threshold=220, minLineLength=150, maxLineGap=35)

    if(lines is None):
        return None

    lines = conversion_to_lines(lines)
    lines = lines_scraping(lines)
    return lines

