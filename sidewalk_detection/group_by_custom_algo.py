import cv2
import numpy as np
from math import dist

dev_mean_thresh = 0.7
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

    ratio_dev_mean = np.std(vec_dist, axis=0)/np.mean(vec_dist, axis=0)

    print(ratio_dev_mean)

    #se il rapporto tra standard deviation e media è inferiore alla soglia, ci sarà un solo gruppo (altrimenti 2)
    if(ratio_dev_mean < dev_mean_thresh):
        return lines, []
    
    #inserisco tutti i punti medi (linee) nei gruppi, in base a quale dei due punti precedenti è più vicino
    i = 0
    groupA = []
    groupB = []
    for med in med_points:
        if dist(A, med) < dist(B, med):
            groupA.append(lines[i])
            frame_test = line.add_line(frame_test, (255,255,255))
        else:
            groupB.append(lines[i])
            frame_test = line.add_line(frame_test)
        i+=1
    

    cv2.imshow("linee divise", cv2.resize(frame_test, (960, 540)))
    cv2.waitKey(0)
    return groupA, groupB