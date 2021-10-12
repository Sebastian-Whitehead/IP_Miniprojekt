import cv2
import numpy as np
from random import *
from collections import deque

#TODO: Adverage colour / blur image
#TODO: Thresholding to identify tile type
#TODO: Crown Identification

randImg = randint(1, 50)
#img = cv2.imread(f"./images/{randImg}.jpg")
img = cv2.imread("./images/1.jpg")
#img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow("Original Image", img)
roi_list = []

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

# masks the center 3/5ths of a list of tiles (ignores houses and such within the image)
def mask_roi_list(img_list):
    masked_roi_list = deque([])     # output array list

    for i in range(len(img_list)):  # loops through input image list
        currentImg = img_list[i]    # Extracts the current Roi
        H = currentImg.shape[0]     # Set width and height as variables to reduce the length of subsequent lines
        W = currentImg.shape[1]

        mask = np.zeros(currentImg.shape[:2], dtype="uint8")        # Sets up blank mask
        cv2.rectangle(mask, (0, 0), (W, H), 255, -1)                # Includes entire frame into mask
        cv2.rectangle(mask, (W//5, H//5), (W-(W//5), H-(H//5)), 0, -1)  # Excludes the center 3/5 of the image

        masked = cv2.bitwise_and(currentImg, currentImg, mask=mask) # Applies mask to current layer
        masked_roi_list.append(masked)   # Appends the masked image to the output list
        '''
        # Displays the Masked Image, The Current region of intrest and the, mask
        cv2.imshow(f"Mask Applied{i}", masked)
        cv2.imshow(f"Current Regoin of Intrest{i}", currentImg)
        cv2.imshow(f"Mask to apply{i}", mask)
        '''

    return masked_roi_list  # Returns a list of masked ROIs

# Averages the colors of the individual tiles
def average_img_color(img_list, input_img):
    assembled_tiles = input_img.copy()
    currentX = 0
    currentY = 0

    for i in range(len(img_list)): # Loop through each individual tile image in img_list

        # Create empty numpy arrays for all unmasked B G R values in an image.
        B = np.array([])
        G = np.array([])
        R = np.array([])

        # CurrentImg = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2HSV)
        currentImg = img_list[i]

        # Loop through all the pixels within the current tile image
        for y, row in enumerate(currentImg):
            for x, pixel in enumerate(row):

                if not pixel[0] == 0: #If the B pixel value is not 0 (only occurs when pixel is masked)

                    # Append B G R values to corresponding list
                    B = np.append(B, pixel[0])
                    G = np.append(G, pixel[1])
                    R = np.append(R, pixel[2])

        # Average and round values of the B G R numpy arrays.
        B = round(np.average(B))
        G = round(np.average(G))
        R = round(np.average(R))
        #print(f"tile - {i} = [{B}, {G}, {R}]")

        # Assembles the array of averaged tiles into a singlar mosaic image.
        if currentX == input_img.shape[1]:
            currentY += input_img.shape[0]//5
            currentX = 0
        #print(f"{currentX} - {currentY}\n")

        cv2.rectangle(assembled_tiles, (currentX, currentY), (currentX + input_img.shape[1]//5, currentY + input_img.shape[0]//5), (B, G, R), -1)
        currentX += input_img.shape[1]//5

    cv2.imshow("test", assembled_tiles)
    return assembled_tiles

# WIP
def threshold_mosaic():
    # Blank matrix for output after thresholding of the mosaic
    ID_Img = [[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]]


tile_list = splitImage(img)
masked_tiles = mask_roi_list(tile_list)
assembled_Mosaic = average_img_color(masked_tiles, img)
cv2.waitKey(0)