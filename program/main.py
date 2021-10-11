import cv2
import numpy as np
from random import *
from collections import deque

#TODO: nested for loops for splitting image
#TODO: Masking to cover center of image
#TODO: Adverage colour / blur image
#TODO: Thresholding to identify tile type
#TODO: Crown Identification

randImg = randint(1, 50)
img = cv2.imread(f"./images/{randImg}.jpg")
cv2.imshow("test", img)
roi_list = []


'''processedImg = [[1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5]]'''

def splitImage(image):
    currentX = 0
    currentY = 0
    roi_list = deque([])
    i = 0
    for y in range(5):
        currentX = 0
        for x in range(5):
            currentROI = image[currentY:(image.shape[0] // 5) + currentY, currentX:(image.shape[1] // 5) + currentX]
            roi_list.append(currentROI)
            #cv2.imshow(f"test{i}", currentROI)
            i += 1
            currentX += img.shape[1] // 5
        currentY += img.shape[0] // 5



splitImage(img)
cv2.waitKey(0)