import os
import cv2
import random
import numpy as np
import tensorflow as tf
import pytesseract
from core.utils import read_class_names
from core.config import cfg
from core.fsm import *

# function to count objects, can return total classes or count per class
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

# function for cropping each detection and saving as new image
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

def infer_tl(img, num_classes, allowed_classes, data):
    colors = dict()
    out_boxes, _, out_classes, num_boxes = data 

    for i in range(num_boxes):
        if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes: continue
        coor = out_boxes[i]
        class_ind = int(out_classes[i])
        class_name = allowed_classes[class_ind]
        if class_name not in 'traffic_light':
            continue
        else: 
            xmin, ymin, xmax, ymax = coor

            cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
            hsv_image = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)

            lower_g = np.array([36,50,20]) 
            upper_g = np.array([75,255,255]) 

            lower_y = np.array([20,50,20])  
            upper_y = np.array([31,255,255]) 

            lower_r = np.array([0,50,20])  
            upper_r = np.array([10,255,255]) 

            gmask = cv2.inRange(hsv_image, lower_g, upper_g)
            ymask = cv2.inRange(hsv_image, lower_y, upper_y)
            rmask = cv2.inRange(hsv_image, lower_r, upper_r)

            color = ['green', 'yellow', 'red']
            pixels = []
            pixels.append(cv2.countNonZero(gmask))
            pixels.append(cv2.countNonZero(ymask))
            pixels.append(cv2.countNonZero(rmask))

            colors[(coor[0], coor[1], coor[2], coor[3])] = color[pixels.index(max(pixels))]

    return colors 

def check_state(data, isSidewalk):
    char = Char()
    curState = char.FSM.curState
    _, _, classes, _ = data

    futureState = curState

    # Check
    if curState == NoState() and isSidewalk : futureState = Walking() 
    if curState == NoState() and 'crosswalk' in classes : futureState = Crossing() 
    if curState == Walking() and 'crosswalk' in classes : futureState = Crossing()
    if curState == Walking() and not isSidewalk and 'crosswalk' not in classes : futureState = NoState()
    if curState == Crossing() and 'crosswalk' not in classes and isSidewalk : futureState = Walking()
    if curState == Crossing() and not isSidewalk and 'crosswalk' not in classes : futureState = NoState()

    # Transition
    if futureState != curState:
        char.FSM.setTransition("to" + futureState.state)
        char.FSM.execute()

# function to run general Tesseract OCR on any detections 
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