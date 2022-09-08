import cv2
import numpy as np
from matplotlib import pyplot as plt

# Creo un'immagine in cui viene mostrata solo la parte dell'immagine originale che mi interessa
def region_of_interest(img, vertices):
    mask                = np.zeros_like(img)
    match_mask_color    = 255
    print(match_mask_color)

    cv2.fillPoly(mask, vertices, match_mask_color)
    mask_img = cv2.bitwise_and(img, mask)

    return mask_img


# Creo i punti delle due rette che disegno
def make_points(image, average):
    slope, y_int = average
    y1 = image.shape[0]
    y2 = int(y1 * (1/2))
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)

    return np.array([x1, y1, x2, y2])


# Serve per ottenere due rette univoche facendo una media tra tutte quelle che trova Hough
def average(image, lines):
    left    = []
    right   = []

    if lines is not None:
        for line in lines:
            print(line)
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


# Semplicemente disegna le rette 
def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 15)
    
    return lines_image


img = cv2.imread('/Users/Edo/Desktop/68_10.png')
copy = np.copy(img)
plt.imshow(copy)
plt.show()

height, width = copy.shape[0], copy.shape[1]

# Creo vertici regione di interesse
region_of_interest_vertices = [
    (0, height - 50),
    (width/2, 300),
    (1200, height)
]

# Applico solite cose di pre-processing
gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blur, 190, 250)

# Ora ho la porzione di immagine che mi interessa 
masked_img = region_of_interest(edges, np.array([region_of_interest_vertices], np.int32))

plt.imshow(masked_img)
plt.show()

# Applico Hough solo alla porzione di immagine per fargli trovare le rette solo in essa
lines = cv2.HoughLinesP(masked_img, rho=1, theta=np.pi/180, threshold=100, lines=np.array([]), minLineLength=100, maxLineGap=10)

averaged_lines  = average(copy, lines)
black_lines     = display_lines(copy, averaged_lines)

lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
lanes = cv2.cvtColor(lanes, cv2.COLOR_BGR2RGB)
plt.imshow(lanes)
plt.show()