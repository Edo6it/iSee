import cv2
import os
from sidewalk_detection.sidewalk_class import line
from sidewalk_detection.sidewalk_class import sidewalk
from sidewalk_detection.sidewalk_class import get_mean_width
from sidewalk_detection.lines_extraction import factory_lines
from sidewalk_detection.left_right_lines import left_right_lines

os.environ['KMP_DUPLICATE_LIB_OK']='True'
dist_thresh = 400

    
def find(frame, side: sidewalk):

    #get the hough lines of the sidewalk
    #frame = cv2.resize(frame, (960, 540))
    lines = factory_lines(frame)

    #compute the left and right lines of the sidewalk
    lineA, lineB = left_right_lines(lines, frame, get_mean_width(side.width_up))

    #extract and deliver the information to the user 
    side.update(lineA, lineB, frame)

    return side




'''img = "C:\\Users\\ilcai\\Videos\\Test_videos\\Video_iPhone5_Test1_Moment.jpg"
frame = cv2.imread(img)
frame = cv2.resize(frame, (960, 540))

side = sidewalk()
side = find(frame, side, 3)

frame = add_sidewalk(side, frame)
cv2.imwrite("total.png", frame)'''
