import cv2
import numpy as np
from matplotlib import pyplot as plt


def region_of_interest(img, vertices):
    mask                = np.zeros_like(img)
    match_mask_color    = 255

    cv2.fillPoly(mask, vertices, match_mask_color)
    mask_img = cv2.bitwise_and(img, mask)

    return mask_img


def make_points(image, average):
    slope, y_int = average
    y1 = image.shape[0]
    y2 = int(y1 * (1/2))
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)

    return np.array([x1, y1, x2, y2])

def average(image, lines):
    left    = []
    right   = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2  = line.reshape(4)
            parameters      = np.polyfit((x1, x2), (y1, y2), 1)
            
            slope           = parameters[0]
            y_int           = parameters[1]
            
            if slope < 0:
                left.append((slope, y_int))
            else:
                right.append((slope, y_int))

    right_avg   = np.average(right, axis=0)
    left_avg    = np.average(left, axis=0)
    left_line   = make_points(image, left_avg)
    right_line  = make_points(image, right_avg)

    return np.array([left_line, right_line])


def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 15)
    
    return lines_image

# img = cv2.imread('/Users/Edo/Desktop/68_10.png')
def process(img):
    copy = np.copy(img)
    
    if copy.shape != ():
        height, width = copy.shape[0], copy.shape[1]

        region_of_interest_vertices = [
            (0, height),
            (width/2, 300),
            (1700, height)
        ]

        gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 190, 250)
        masked_img = region_of_interest(edges, np.array([region_of_interest_vertices], np.int32))

        lines = cv2.HoughLinesP(masked_img, rho=1, theta=np.pi/180, threshold=100, lines=np.array([]), minLineLength=100, maxLineGap=10)

        averaged_lines  = average(copy, lines)
        black_lines     = display_lines(copy, averaged_lines)

        lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
        return lanes

cap = cv2.VideoCapture("/Users/Edo/Desktop/video_clip.mp4")

while(cap.isOpened()):
    ret, frame  = cap.read()
    frame       = process(frame)
    
    if frame is not None:
        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
