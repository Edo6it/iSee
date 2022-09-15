import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# this method manages the convertion from csv b-box to yolo format
def bndBox2YoloLine(self, box, classList=[]):
    xMin = box['Upper left corner X']
    xMax = box['Lower rigth corner X']
    yMin = box['Upper left corner Y']
    yMax = box['Lower right corner Y']

    xCen = float((xMin + xMax)) / 2 / self.imgSize[1]
    yCen = float((yMin + yMax)) / 2 / self.imgSize[0]

    w = float((xMax - xMin)) / self.imgSize[1]
    h = float((yMax - yMin)) / self.imgSize[0]

    boxName = box['Annotation tag']
    if boxName not in classList:
        classList.append(boxName)

    classIndex = classList.index(boxName)

    return classIndex, xCen, yCen, w, h


def main(self):
    df = pd.read_csv('/Volumes/SANDISK/Video CV/traffic_lights/Annotations/Annotations/daySequence1/frameAnnotationsBOX.csv', sep=';')

    classIndex, xCen, yCen, w, h = bndBox2YoloLine()
    yolo_df = pd.DataFrame([classIndex, xCen, yCen, w, h])

if __name__ == '__main__':
    main()

