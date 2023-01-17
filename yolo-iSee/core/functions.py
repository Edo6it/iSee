import os
import cv2
import random
import numpy as np
import tensorflow as tf
import pytesseract
from core.utils import read_class_names
from core.config import cfg
from core.fsm import *
from sidewalk_detection.sidewalk_main import find
from sidewalk_detection.sidewalk_class import sidewalk
# ==============================================================

# PEOPLE COUNTER

def count_objects(data, by_class = True):
    boxes, scores, classes, num_objects = data

    #create dictionary to hold count of objects
    counts = dict()

    # if by_class = True then count objects per class
    if by_class:
        class_names = read_class_names(cfg.YOLO.CLASSES)

        # loop through total number of objects found
        for i in range(num_objects):
            # grab class index and convert into corresponding class name
            class_index = int(classes[i])
            class_name = class_names[class_index]
            if class_name in 'person':
                counts[class_name] = counts.get(class_name, 0) + 1
            else:
                continue

    # else count total objects found
    else:
        counts['total object'] = num_objects
    
    return counts

# ==============================================================

# CROP IMAGES DETECTIONS 

def crop_objects(img, data, path, allowed_classes):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    #create dictionary to hold count of objects for image name
    counts = dict()
    for i in range(num_objects):
        # get count of class for part of image name
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes:
            counts[class_name] = counts.get(class_name, 0) + 1
            # get box coords
            xmin, ymin, xmax, ymax = boxes[i]
            # crop detection from image (take an additional 5 pixels around all edges)
            cropped_img = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
            # construct image name and join it to path for saving crop properly
            img_name = class_name + '_' + str(counts[class_name]) + '.png'
            img_path = os.path.join(path, img_name )
            # save image
            cv2.imwrite(img_path, cropped_img)
        else:
            continue

# ==============================================================

# STATE TRAFFIC LIGHTS (red, yellow, green)

def crop_region(cropped_img):
    #togliamo un ulteriore 3% di immagine da ogni lato del bounding box
    crop_height = cropped_img.shape[0]
    crop_width = cropped_img.shape[1]
    crop_frame = cropped_img[int(crop_height*0.08):int(crop_height - (crop_height*0.08)), int(crop_width*0.08):int(crop_width- (crop_width*0.08))]
    #print(crop_height, crop_width, crop_frame.shape[0], crop_frame.shape[1])

    #Divido in 3 regioni, una per ogni colore
    region_height = int(crop_frame.shape[0]/3)
    region_width = crop_frame.shape[1]
    #print(region_height)
    reg1 = crop_frame[0:region_height, 0:region_width]
    
    reg2 = crop_frame[region_height+1:region_height*2, 0:region_width]
    
    reg3 = crop_frame[region_height*2+1:crop_frame.shape[0], 0:region_width]
    
    return reg1, reg2, reg3


def check_tl(img, num_classes, allowed_classes, data):
    colors = dict()
    out_boxes, _, out_classes, num_boxes = data 
    j=0

    for i in range(num_boxes):
        if int(out_classes[i]) < 0 or int(out_classes[i]) >= num_classes: continue  #PENSO SIA >=
        coor = out_boxes[i]
        class_ind = int(out_classes[i])
        class_name = allowed_classes[class_ind]

        if class_name not in 'traffic_light':
            continue
        else: 
            xmin, ymin, xmax, ymax = coor

            cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
            
            #dividiamo l'immagine in 3 regioni
            reg1, reg2, reg3 = crop_region(cropped_img)
         
            #hsv_image = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)
            hsv_reg1 = cv2.cvtColor(reg1, cv2.COLOR_BGR2HSV)
            hsv_reg2 = cv2.cvtColor(reg2, cv2.COLOR_BGR2HSV)
            hsv_reg3 = cv2.cvtColor(reg3, cv2.COLOR_BGR2HSV)

            #Definisco gli intervalli per i colori
            lower_g = np.array([36,50,70]) 
            upper_g = np.array([79,255,255]) 

            lower_y = np.array([26, 50, 70])  
            upper_y = np.array([31,255,255]) 

            lower_r = np.array([0, 50, 70])  
            upper_r = np.array([23, 255, 255]) 

            lower_r2 = np.array([160,100,20])
            upper_r2 = np.array([179,255,255])

            lower_w = np.array([0,0,180])
            upper_w = np.array([255,255,255])

            gmask = cv2.inRange(hsv_reg3, lower_g, upper_g)
            ymask = cv2.inRange(hsv_reg2, lower_y, upper_y)
            rmask = cv2.inRange(hsv_reg1, lower_r, upper_r)
            r2mask = cv2.inRange(hsv_reg1, lower_r2, upper_r2) 

            gmask_w = cv2.inRange(hsv_reg3, lower_w, upper_w)
            ymask_w = cv2.inRange(hsv_reg2, lower_w, upper_w)
            rmask_w = cv2.inRange(hsv_reg1, lower_w, upper_w) 

            color = ['green', 'yellow', 'red']
            count_g = cv2.countNonZero(gmask)
            count_y = cv2.countNonZero(ymask)
            count_r = cv2.countNonZero(rmask)+cv2.countNonZero(r2mask)
            pixels = []
            if (count_g > 0):
                pixels.append(count_g+cv2.countNonZero(gmask_w))
            else:
                pixels.append(count_g)
            if (count_y > 0):
                pixels.append(count_y+cv2.countNonZero(ymask_w))
            else:
                pixels.append(count_y)
            if (count_r > 0):
                pixels.append(count_r+cv2.countNonZero(rmask_w))
            else:
                pixels.append(count_r)
            #print(pixels)

            if (max(pixels)>35):
                colors[j] = color[pixels.index(max(pixels))]
                j += 1
    return colors

# ==============================================================

# FSM
char = Char()

def check_state(isSidewalk, isCrossing, tl_colors, state):
    curState = state
    futureState = curState 

    # Check
    if curState == State.NoState.value and isSidewalk : futureState = State.Walking.value
    if curState == State.NoState.value and isCrossing : futureState = State.Crossing.value

    if curState == State.Walking.value and isCrossing : futureState = State.Crossing.value
    if curState == State.Walking.value and not isSidewalk and isCrossing : futureState = State.NoState.value

    if curState == State.Crossing.value and not isCrossing and isSidewalk : futureState = State.Walking.value
    if curState == State.Crossing.value and not isSidewalk and not isCrossing : futureState = State.NoState.value 
    if curState == State.Crossing.value and not tl_colors : futureState = State.CrossingNoTl.value

    if curState == State.CrossingNoTl.value and tl_colors : futureState = State.Crossing.value
    if curState == State.CrossingNoTl.value and not isCrossing and isSidewalk : futureState = State.Walking.value
    if curState == State.CrossingNoTl.value and not isSidewalk and not isCrossing : futureState = State.NoState.value 

    return futureState

def perform_transition(history):
    counter = {'NoState':0, 'Walking':0, 'Crossing':0, 'CrossingNoTl':0}

    for state in history:
        if state in State.Walking.value:
            counter['Walking'] += 1
        elif state in State.Crossing.value:
            counter['Crossing'] += 1
        elif state in State.CrossingNoTl.value:
            counter['CrossingNoTl'] += 1
        else:
            counter['NoState'] += 1

    futureState = max(counter, key=counter.get)

    char.FSM.setTransition("to" + futureState)
    char.FSM.execute()

def perform_action(distance, tl_colors, isCars):
    state = char.FSM.curState

    if state == "Walking":
        if distance:
            print(f"Striscia individuata a {distance}")
    
    elif state == "Crossing":
        if tl_colors:
            print(f"Il semaforo è {tl_colors[0]}")

    elif state == "CrossingNoTL":
        if isCars:
            print("Macchina in arrivo, attendere\n \
                Guarda a sx e a dx")
        else:
            print("Attraversa")
        
# ==============================================================

# CHECK CROSSWALK AND ITS DISTANCE
def check_crossing(data, center):
    th = 250000.0
    cross = False  
    distance = 0
    boxes, scores, classes, num_objects = data
    min_distance = 20
    indeces = np.where(classes[:num_objects] == 1.)[0]
    cross_boxes, cross_scores = [boxes[i] for i in indeces.tolist()], [scores[i] for i in indeces.tolist()]
    f = 1660
    crosswalk_width = 3

    for i in range(len(indeces.tolist())):
        if cross_scores[i] < 0.6:
            continue 

        xmin, ymin, xmax, ymax = cross_boxes[i]

        p1 = np.array((xmin, ymin))
        p2 = np.array((xmax, ymin))
        p3 = np.array((xmin, ymax))

        h = np.linalg.norm(p1 - p3)
        w = np.linalg.norm(p1 - p2)

        area = h * w 

        bbox_center = np.array(((xmin+xmax) // 2, (ymin+ymax) // 2))

        if (bbox_center[0] >= center[0] - 400 and bbox_center[0] <= center[0] + 400) and bbox_center[1] > center[1]:
            cross = True 

        if area > th and cross:
            # the crosswalk is in front of me
            # this is the crosswalk we want to take into account, thus we ignore the others 
            break
        else:
            # the crosswalk is perpendicular to the road
            # compute the distance with the crosswalks we found and keep the closest one 
            distance = (f * crosswalk_width) / w
            print(f'\nDISTANCE: {distance}\n')
            if distance < min_distance:
                min_distance = distance

    return cross, distance 

# ==============================================================

# CHECK SIDEWALK
def add_sidewalk(self: sidewalk, frame, color_left=(0,0,0), color_right=(255,255,255), size=20):
    if(self.left_line.slope is None and self.right_line.slope is None):
        return frame

    frame = cv2.line(frame, self.left_line.point_low, self.left_line.point_up, color_left, size, cv2.LINE_AA)
    frame = cv2.line(frame, self.right_line.point_low, self.right_line.point_up, color_right, size, cv2.LINE_AA)
    frame = cv2.line(frame, self.mid_line.point_low, self.mid_line.point_up, (120,60,90), size, cv2.LINE_AA)
    return frame


def information(side: sidewalk, state):
    #get the slope of the line, to understand if the sidewalk is straight or turning (left or right)
    #print(side.mid_line.slope)
    if(side.mid_line.slope is None or state != "Walking"):
        return side

    if(side.count >= 60):
        if(side.mid_line.slope > -1.2 and side.mid_line.slope < 0):
            print("Il marciapiede curva verso destra")
            side.count = 0
        elif(side.mid_line.slope > 0 and side.mid_line.slope < 1.2):
            print("Il marciapiede curva verso sinistra")
            side.count = 0
        
        #get position of the sidewalk relatively to the person perspective
        point_left = side.left_line.mean_point()
        point_right = side.right_line.mean_point()
        reference_point = (960, 540)
        if(point_right[0] < reference_point[0]):
            print("sei alla destra del marciapiede, accentrati verso sinistra")
            side.count = 0
        elif(point_left[0] > reference_point[0]):
            print("sei alla sinistra del marciapiede, accentrati verso destra")
            side.count = 0
    else:
        side.count += 1

    return side



def check_sidewalk(frame, side: sidewalk, state):
    side = find(frame, side)
    side = information(side, state)

    if(side.mid_line.slope is None):
        return False, side
    else:
        return True, side

# ==============================================================

# CHECK CARS 
def check_cars(num_classes, allowed_classes, data):
    _, _, out_classes, num_boxes = data 
    car_presence = False
    for i in range(num_boxes):
        if int(out_classes[i]) < 0 or int(out_classes[i]) >= num_classes: continue 
        class_ind = int(out_classes[i])
        class_name = allowed_classes[class_ind]
        if class_name not in 'car':
            continue    
        else:
            car_presence = True
            break
    return car_presence

# ==============================================================

# TESSERACT OCR 

def ocr(img, data):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    for i in range(num_objects):
        # get class name for detection
        class_index = int(classes[i])
        class_name = class_names[class_index]
        # separate coordinates from box
        xmin, ymin, xmax, ymax = boxes[i]
        # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
        box = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
        # grayscale region within bounding box
        gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
        # threshold the image using Otsus method to preprocess for tesseract
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # perform a median blur to smooth image slightly
        blur = cv2.medianBlur(thresh, 3)
        # resize image to double the original size as tesseract does better with certain text size
        blur = cv2.resize(blur, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
        # run tesseract and convert image text to string
        try:
            text = pytesseract.image_to_string(blur, config='--psm 11 --oem 3')
            print("Class: {}, Text Extracted: {}".format(class_name, text))
        except: 
            text = None