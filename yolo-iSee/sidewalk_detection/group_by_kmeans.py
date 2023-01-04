import cv2
from sklearn.cluster import KMeans

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
        
    
    #applico kmeans sui punti medi (2 gruppi)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(med_points)

    #inserisco tutti i punti medi (linee) nei gruppi, in base alle label dell'algoritmo kmeans
    i = 0
    for line in lines:
        if kmeans.labels_[i] == 0:
            groupA.append(line)
            frame_test = line.add_line(frame_test, (255,255,255))
        else:
            groupB.append(line)
            frame_test = line.add_line(frame_test)
        i+=1
    
    cv2.imshow("linee divise", cv2.resize(frame_test, (960, 540)))
    cv2.waitKey(0)
    return groupA, groupB