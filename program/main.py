import cv2
import numpy as np
from random import *
from collections import deque
import math

next_id = 10 # Next id variable for grassfire() function

# Choose a random picture from the test set and load the selected image
randImg = randint(1, 50)
img = cv2.imread(f"./images/{randImg}.jpg")
print(f"Random image to process = ./images/{randImg}.jpg")

# Show the original image before processing
cv2.imshow("Original Image", img)


# splits image into individual tiles in a 5x5 grid
def split_image(image):
    currentY = 0  # Y within the image
    #i = 0

    roi_list = deque([])  # Initialize the output array

    for y in range(5):  # Loop through rows.
        currentX = 0  # Reset X position every time the program switches rows
        for x in range(5):  # Loop through columns

            # Crop out 1/25th of the original image.
            currentROI = image[currentY:(image.shape[0] // 5) + currentY, currentX:(image.shape[1] // 5) + currentX]

            roi_list.append(currentROI)  # Append the cropped tile to a the output list

            #i += 1
            #cv2.imshow(f"test{i}", currentROI)


            currentX += img.shape[1] // 5  # Shift the column section to crop
        currentY += img.shape[0] // 5  # Shift the row section to crop
    return roi_list  # Return the ROI list


# masks the center 3/5ths of a list of tiles (ignores houses and such within the image)
def mask_roi_list(img_list):
    masked_roi_list = deque([])  # output array list

    for i in range(len(img_list)):  # loops through input image list
        currentImg = img_list[i]  # Extracts the current Roi
        H = currentImg.shape[0]  # Set width and height as variables to reduce the length of subsequent lines
        W = currentImg.shape[1]

        mask = np.zeros(currentImg.shape[:2], dtype="uint8")  # Sets up blank mask
        cv2.rectangle(mask, (0, 0), (W, H), 255, -1)  # Includes entire frame into mask
        cv2.rectangle(mask, (W // 5, H // 5), (W - (W // 5), H - (H // 5)), 0,
                      -1)  # Excludes the center 3/5 of the image

        masked = cv2.bitwise_and(currentImg, currentImg, mask=mask)  # Applies mask to current layer
        masked_roi_list.append(masked)  # Appends the masked image to the output list

        '''
        # Displays the Masked Image, The Current region of interest and the, mask
        cv2.imshow(f"Mask Applied{i}", masked)
        cv2.imshow(f"Current ROI - {i}", currentImg)
        cv2.imshow(f"Mask to apply{i}", mask)
        '''

    return masked_roi_list  # Returns a list of masked ROIs


# Averages the colors of the individual tiles and assembles them into a mosaic
def average_img_color(img_list, input_img):
    assembled_tiles = input_img.copy()
    currentX = 0
    currentY = 0

    for i in range(len(img_list)):  # Loop through each individual tile image in img_list

        # Create empty numpy arrays for all unmasked B G R values in an image.
        B = np.array([])
        G = np.array([])
        R = np.array([])

        currentImg = img_list[i]

        # Loop through all the pixels within the current tile image
        for y, row in enumerate(currentImg):
            for x, pixel in enumerate(row):

                if not pixel[0] == 0:  # If the B pixel value is not 0 (only occurs when pixel is masked)

                    # Append B G R values to corresponding list
                    B = np.append(B, pixel[0])
                    G = np.append(G, pixel[1])
                    R = np.append(R, pixel[2])

        # Average and round values of the B G R numpy arrays.
        B = round(np.average(B))
        G = round(np.average(G))
        R = round(np.average(R))
        # print(f"tile - {i} = [{B}, {G}, {R}]")

        # Assembles the array of averaged tiles into a singular mosaic image.
        if currentX == input_img.shape[1]:
            currentY += input_img.shape[0] // 5
            currentX = 0
        # print(f"{currentX} - {currentY}\n")

        # Draw a rectangle matching the size of the tile onto a blank image
        cv2.rectangle(assembled_tiles, (currentX, currentY),
                      (currentX + input_img.shape[1] // 5, currentY + input_img.shape[0] // 5), (B, G, R), -1)
        currentX += input_img.shape[1] // 5

    cv2.imshow("Mosaic", assembled_tiles)
    return assembled_tiles


# HSV thresholding of the assembled mosaic
def threshold_mosaic(input_mosaic):
    input_mosaic = cv2.cvtColor(input_mosaic, cv2.COLOR_BGR2HSV)
    # cv2.imshow("Mosaic_HSV", Input_mosaic)

    # Blank matrix for output after thresholding of the mosaic
    ID_Img = np.zeros((5, 5), dtype="uint8")
    # 0 = Unknown
    # 1 = Grass
    # 2 = Ocean
    # 3 = Field
    # 4 = Forest
    # 5 = Swamp
    # 6 = Mine

    # x and y offset so loop samples center of images
    x_offset = 50
    y_offset = 50

    for y, row in enumerate(ID_Img):
        for x, entry in enumerate(row):
            test_Value = input_mosaic[y_offset + y * (input_mosaic.shape[0] // 5), x_offset + x * (
                    input_mosaic.shape[1] // 5)]  # sample the center of each tile
            H = test_Value[0]  # Extract the HSV values individually
            S = test_Value[1]
            V = test_Value[2]
            # print(f"{x},{y} - {test_Value}")

            # threshold the obtained HSV value and id the tile in the output array.
            if (36 <= H <= 47) and (135 <= S <= 235) and (90 <= V <= 160):  # Grass
                ID_Img[y, x] = 1

            elif (103 <= H <= 110) and (110 <= S <= 260) and (100 <= V <= 195):  # Water
                ID_Img[y, x] = 2

            elif (23 <= H <= 28) and (190 <= S <= 250) and (85 <= V <= 205):  # Field
                ID_Img[y, x] = 3

            elif (30 <= H <= 88) and (60 <= S <= 220) and (40 <= V <= 90):  # Forest
                ID_Img[y, x] = 4

            elif (19 <= H <= 28) and (50 <= S <= 165) and (70 <= V <= 140):  # Swamp
                ID_Img[y, x] = 5

            elif (20 <= H <= 32) and (10 <= S <= 165) and (30 <= V <= 65):  # Mine
                ID_Img[y, x] = 6

            # if ID_Img[y, x] == 0:
            #    print(f" Tile [{x}, {y}] : [{H}, {S}, {V}]")

    # TODO: ERROR -
    # Image 24, 27, 31 Invalid tile within thresholds
    # Image 25, 29 not square
    # Image 14 Large Castle Overlap

    print(f"\n🔍 Identified tiles =\n{ID_Img}")
    return ID_Img


# Grouping the identified mosaic
def ignite_fire(mosaic, coordinates, current_id, target_tile):
    # Create burn_queue deque to keep track of positions to burn
    burn_queue = deque([])
    something_burned = False

    # If object pixel, add starting point to deque
    if mosaic[coordinates[0], coordinates[1]] == target_tile:
        burn_queue.append(coordinates)
        something_burned = True

    # While there are still items in the burn queue
    while len(burn_queue) > 0:
        current_pos = burn_queue.pop()  # Pop last tile from queue
        y, x = current_pos
        # Burn current_pos with current id
        mosaic[y, x] = current_id

        # Add connections to burn_queue
        if y - 1 >= 0 and mosaic[y - 1, x] == target_tile:
            burn_queue.append((y - 1, x))
        if x - 1 >= 0 and mosaic[y, x - 1] == target_tile:
            burn_queue.append((y, x - 1))
        if y + 1 < mosaic.shape[0] and mosaic[y + 1, x] == target_tile:
            burn_queue.append((y + 1, x))
        if x + 1 < mosaic.shape[1] and mosaic[y, x + 1] == target_tile:
            burn_queue.append((y, x + 1))
    if something_burned:
        current_id += 1

    return current_id, mosaic


# Function for detecting crowns and dividing them into a 5x5 matrix
def detect_crowns():
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to greyscale
    cv2.imshow("greyscale",img_grey)
    crown_list = []  # Declare empty list for crown coordinates

    # template match using the standard crown template
    temp = cv2.imread('crownTemplate.jpg', 0)
    crown_list = template_search(crown_list, img_grey, temp, 0.71)

    # template match using the field crown template (fix because it didn't work with the other one for some reason)
    temp = cv2.imread('fieldCrownTest.jpg', 0)
    crown_list = template_search(crown_list, img_grey, temp, 0.64)

    # Assign the according to location to a matrix of similar size and shape as the ID mosaic
    crown_matrix = np.zeros((5, 5), dtype="uint8")
    for crown in crown_list:
        crown_matrix[int(crown[1] / 100), int(crown[0] / 100)] += 1

    # Console print
    print(f"\n 👑 {len(crown_list)} : Total crowns found")
    print(crown_matrix)

    return crown_matrix


# Function for searching an image with a given template and threshold
def template_search(crowns, source, template, threshold):
    for i in range(4):  # Repeat for the 4 possible rotations of the template image
        template = cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)  # rotate the template image 90 degrees clockwise
        # cv2.imshow(f"test{i}", temp)
        w, h = template.shape[::-1]  # Set the width and height for the box to draw around the discovered crowns

        res = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)  # template match using TM_CCOEFF_NORMED
        cv2.imshow("res", res)  # Show the template matching result
        loc = np.where(res >= threshold)  # Find all locations on the tested image with a value greater than threshold

        # prevent multiple positive tests for crowns within 5 pixels of each other
        check = False
        for pt in zip(*loc[::-1]):
            for [x, y] in crowns:
                check = False
                if (math.isclose(x, pt[0], abs_tol=10)) and (math.isclose(y, pt[1], abs_tol=10)):
                    check = True
                    # print(f"{pt} - ({x}, {y}) >>>>> {abs(x - pt[0])} - {abs(y - pt[1])} -- TOO CLOSE")
                    break
                else:
                    # print(f"{pt} - ({x}, {y}) >>>>> {abs(x - pt[0])} - {abs(y - pt[1])}")
                    continue

            # If there are no previously detected crowns near by then append to the crowns list
            if not check:
                crowns.append(pt)
                cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imshow("crowns found", img)  # Display an image with red rectangles drawn around the detected crowns
    cv2.imwrite('res.jpg', img)
    return crowns


# Calculate the score for the provided board from the group matrix and the 5x5 crown matrix
def calculate_score(id_matrix, crown_matrix):
    final_score = 0

    # Repeat for each valid group on the board
    for id in range((np.amax(id_matrix) - 10) + 1):
        #print(f"\n>>> {id + 10}")
        group_crowns = 0
        property_size = 0

        # Split the group id matrix into rows
        for y in range(5):
            x = np.where(id_matrix[y] == id + 10)[0]  # find all values of the given id within the current row
            property_size += len(x)  # Sum the property size of a given group
            for n in x:
                # print(f"({n},{y}) - {Crown_matrix[y, n]}")

                # check each property tile for a crown and sum the total in the property
                group_crowns += crown_matrix[y, n]

        #print(f"{property_size} * {group_crowns} = {property_size * group_crowns}")
        final_score += property_size * group_crowns # Calculate the final score for the board

    print(f"\n⭐ SCORE = {final_score} ⭐\n")
    return final_score


# ----------------------------------------------------------------------------------------------- #
tile_list = split_image(img)
masked_tiles = mask_roi_list(tile_list)
assembled_Mosaic = average_img_color(masked_tiles, img)
ID_mosaic = threshold_mosaic(assembled_Mosaic)

print("-----------------------------------------------------")

for i in range(6):
    for y, row in enumerate(ID_mosaic):
        for x, pixel in enumerate(row):
            # For each pixel call ignite_fire(…)
            next_id, ID_mosaic = ignite_fire(ID_mosaic, (y, x), next_id, i + 1)
print(f"\n🔥 Grouped Mosaic =\n{ID_mosaic}")

print("-----------------------------------------------------")

Crown_mosaic = detect_crowns()

print("-----------------------------------------------------")

calculate_score(ID_mosaic, Crown_mosaic)

cv2.waitKey(0)