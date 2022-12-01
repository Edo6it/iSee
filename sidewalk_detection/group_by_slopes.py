import cv2

dist_thresh = 400



def get_groups(lines: list, frame_test):
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
    for line in lines:
        if (line.slope > 0):
            groupA.append(line)
            frame_test = line.add_line(frame_test, (255,255,255))
        else:
            groupB.append(line)
            frame_test = line.add_line(frame_test)
        
    
   
    
    cv2.imshow("linee divise", cv2.resize(frame_test, (960, 540)))
    cv2.waitKey(0)
    return groupA, groupB