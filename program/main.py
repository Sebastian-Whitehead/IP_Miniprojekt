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

# splits image into individual tiles in a 5x5 grid
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
    return roi_list

def mask_roi_list(img_list):
    masked_roi_list = deque([])     # output array list

    for i in range(len(img_list)):  # loops through input image list
        currentImg = img_list[i]    # Extracts the current Roi
        H = currentImg.shape[0]     # Set width and height as varriables to reduce the length of subsequent lines
        W = currentImg.shape[1]

        mask = np.zeros(currentImg.shape[:2], dtype="uint8")        # Sets up blank mask
        cv2.rectangle(mask, (0, 0), (W, H), 255, -1)                # Incluedes entire frame into mask
        cv2.rectangle(mask, (W//5, H//5), (W-(W//5), H-(H//5)), 0, -1)  # Exludes the center 3/5 of the image

        masked = cv2.bitwise_and(currentImg, currentImg, mask=mask) # Applies mask to current layer
        masked_roi_list.append(masked)   # Appends the masked image to the output list

        ''' # Displays the Masked Image, The Current region of intrest and the, mask
        cv2.imshow(f"Mask Applied{i}", masked)
        cv2.imshow(f"Current Regoin of Intrest{i}", currentImg)
        cv2.imshow(f"Mask to apply{i}", mask)
        '''

    return masked_roi_list  # Returns a list of masked ROIs







tile_list = splitImage(img)
output = mask_roi_list(tile_list)
cv2.waitKey(0)