import cv2
import os
from sidewalk_class import line
from sidewalk_class import sidewalk
from lines_extraction import factory_lines
from left_right_lines import left_right_lines

os.environ['KMP_DUPLICATE_LIB_OK']='True'
img_path = "C:\\Users\\ilcai\\Pictures\\DatasetTest\\GoProModified\\VideoGoPro1_corrected\\GoPro1_1_Corrected.png"
img_test = "C:\\Users\\ilcai\\Pictures\\DatasetTest\\VideoIPhone1\\iPhone_1.png"
dist_thresh = 400



def info_extract(frame, lineA: line, lineB: line, side: sidewalk):
    if(lineA is None and lineB is None):
        print("impossibile identificare marciapiede")
        return
    
    if(lineB is None):
        print("solo una linea trovata!")
        return

    print("entrambe le linee trovate!")
    
    #update of sidewalk model
    side.update(lineA, lineB, frame)
    side.information()
    
    #show the line on screen
    frame = side.add_sidewalk(frame)
    cv2.imshow("Hough_lines", cv2.resize(frame, (960, 540), interpolation = cv2.INTER_AREA))
    cv2.waitKey(0)

    return side
    

def find(frames):
    side = sidewalk()

    for frame in frames:
        #get the hough lines of the sidewalk
        lines = factory_lines(frame)

        #compute the left and right lines of the sidewalk
        lineA, lineB = left_right_lines(lines, frame_test)

        #extract and deliver the information to the user 
        side = info_extract(frame, lineA, lineB, side)
            


frames = []
frame = cv2.imread(img_path)
frame_test = cv2.imread(img_path)
frames.append(frame)
find(frames)



#connected component labeling (prima bisogna applicare trasformata e fare preprocessing)