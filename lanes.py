# Mikail Usman • 2/6/2025
# Real Time Lane Detection System (RTLDS)

import cv2 # From full package [opencv-contrib-python]
import numpy as np
import matplotlib.pyplot as plt

# -- Accessing and Applying Initial Processing -- 
def imageProcessor(inputIMG):
    image = cv2.imread(inputIMG) # Reads image and returns it as a multi-dimensional numPy array containing pixel intensities.
    laneImageCopy = np.copy(image) # Copying image array || Keep original separate from the one we will process.
    imageGS = cv2.cvtColor(laneImageCopy, cv2.COLOR_RGB2GRAY) # Converting image (laneImageCopy) to grayscale. 
    imageBlur = cv2.GaussianBlur(imageGS, (5,5), 0) # Gaussian blur to reduce noise w/ 5x5 Kernel and deviation of 0.
    imageCanny = cv2.Canny(imageBlur, 50, 150) # Executing Canny function with thresholds b/w 50 and 150.
    maskedIMGFinal = polygonMask(imageCanny) # Executing polygonMask() to outline region of interest.

    return laneImageCopy, maskedIMGFinal

# -- Creating a Mask Outlining Region of Interest -- 
def polygonMask(image):
    height = image.shape[0] # Getting height (no.of rows) using shape of the image array.
    # Creating a polygon to cover the region of interest (Lane):
    polygons = np.array([
        [(200, height), (1100, height), (500, 250)] # Uses coordinates of lane zone from image (courtesy of pyplot).
    ])
    # Creates black mask over image - by making an array of zeros:
    # || Creates black image with same dimensions as original image
    mask = np.zeros_like(image) # (Each value = pixel intensity of zero I.E Black) with the same no. of rows/cols as OG image.
    cv2.fillPoly(mask, polygons, 255) # Area bounded by the polygonal contour will be fully white.
    maskedImage  = cv2.bitwise_and(image, mask) # Bitwise AND Operator to compare Image, Mask and only show region covered by white polygon.
    return maskedImage

# -- Drawing Lines Generated from Hough Transform -- 
def displayMarkings(image, lines):
    linesMask = np.zeros_like(image) # Creates black image with same dimensions as original image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4) # Line array unpacked into 4 coordinates of start and end positions
            # On an image, draws a line segment by connecting two points with RGB color and thickness. 
            cv2.line(linesMask, (x1, y1), (x2,y2), (255, 0, 0), 10)
    return linesMask # Returns black image but with line markings.

# -- Combining Generated Lane Lines and Original Image
def mergeResult(image):
    laneImageCopy, maskedIMGFinal = image

    # Hough Transform via func(image, pixel precision for bin size, theta precision for bin size, threshold)
    createLineMarkings = cv2.HoughLinesP(maskedIMGFinal, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) 
    lineImage = displayMarkings(laneImageCopy, createLineMarkings) # Displaying generated lane markings onto OG media.
    mergedImage = cv2.addWeighted(laneImageCopy, 0.8, lineImage, 1, 1) # Take weighted sum of OG image array and lineImage array to find resultant merged image.
    return mergedImage

if __name__ == "__main__":
    testIMG = imageProcessor('/Users/m20mi/Documents/Work/RT Lanes/test_image.jpg')
    cv2.imshow('Display Test', mergeResult(testIMG)) # Display image
    cv2.waitKey(0) # Displays image until <any> key is pressed. 
    #plt.imshow(testIMG)
    #plt.show()



