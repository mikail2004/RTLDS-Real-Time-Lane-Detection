# Mikail Usman • 2/6/2025
# Real Time Lane Detection System (RTLDS)

import cv2 # From full package [opencv-contrib-python]
import numpy as np
import matplotlib.pyplot as plt

# -- Accessing and Applying Initial Processing -- 
def imageProcessor(inputIMG):
    image = cv2.imread(inputIMG) # Reads image and returns it as a multi-dimensional numPy array containing pixel intensities.
    laneImageCopy = np.copy(image) # Copying image array || Keep original separate from the one we will process.
    imageGS = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # Converting image (laneImageCopy) to grayscale. 
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
            x1, y1, x2, y2 = line # Line array unpacked into 4 coordinates of start and end positions
            # On an image, draws a line segment by connecting two points with RGB color and thickness. 
            cv2.line(linesMask, (x1, y1), (x2,y2), (255, 0, 0), 10)
    return linesMask # Returns black image but with line markings.

# -- Optimizating by Averaging Line Markers --
def avgIntercept(image, lines):
    # There are multiple lines for the right side of the lane and the left side
    # Record the slopes and intercepts (m and c) of each line depending on the side as a list
    leftFit = [] # Slopes and intercepts (m and c) of the left side of the lane
    rightFit = [] # Slopes and intercepts (m and c) of the left side of the lane
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4) # Unpacking coordinates of each line
        # Polynomial of degree 1, Uses coordinates of a line to return vector describing slope and y-intercept
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0] # As slope is at the first index of the returned vector <parameters>
        intercept = parameters[1]

        # To see if the slope/intercept returned is of the right side of the lane or left.
        # Lines on the left will have a negative slope (m<0)
        # Lines on the right will have a positive slope (m>0)
        if slope < 0: 
            leftFit.append((slope, intercept))
        else:
            rightFit.append((slope, intercept))

    if leftFit:
        leftFitAvg = np.average(leftFit, axis=0)
        print(leftFitAvg, 'left')
        leftLine = calcCoords(image, leftFitAvg)
    if rightFit:
        rightFitAvg = np.average(rightFit, axis=0)
        print(rightFitAvg, 'right')
        rightLine = calcCoords(image, rightFitAvg)

    # # Average out all intercepts and slopes of each side of the lane:
    # leftFitAvg = np.average(leftFit, axis=0)
    # rightFitAvg = np.average(rightFit, axis=0)

    # # Combining intercepts and slopes with coordinates to create a proper line (y=mx+c)
    # leftLine = calcCoords(image, leftFitAvg)
    # rightLine = calcCoords(image, rightFitAvg)

    return np.array([leftLine, rightLine])

# -- Calculate Coordinates for Optimization Averaging Function --
def calcCoords(image, lineParameters):
    # Finding Coordinates needed to draw lane line using averaged out slope and intercept (y=mx+c)
    slope, intercept = lineParameters
    y1 = image.shape[0] # Fixing minimum height to 0 (starting point of the image height).
    y2 = int(y1*3/5) # Fixing maximum height (based on image height) to be about 3/5th the image height.
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

# -- Combining Generated Lane Lines and Original Image
def mergeResult(image):
    laneImageCopy, maskedIMGFinal = image

    # Hough Transform via func(image, pixel precision for bin size, theta precision for bin size, threshold)
    createLineMarkings = cv2.HoughLinesP(maskedIMGFinal, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) 
    averageLineMarkings = avgIntercept(laneImageCopy, createLineMarkings) # Average out multiple markings into one for each lane side.
    lineImage = displayMarkings(laneImageCopy, averageLineMarkings) # Displaying generated lane markings onto OG media.
    mergedImage = cv2.addWeighted(laneImageCopy, 0.8, lineImage, 1, 1) # Take weighted sum of OG image array and lineImage array to find resultant merged image.
    return mergedImage

def videoProcessor(videoSRC):
    capture = cv2.VideoCapture(videoSRC) # Receives video from a link
    # If video capture is active, start loop
    while(capture.isOpened()): 
        _, frame = capture.read() # First value is a boolean, the second is a frame of the video.
        imageCanny = cv2.Canny(frame, 50, 150)
        maskedIMGFinal = polygonMask(imageCanny) 
        createLineMarkings = cv2.HoughLinesP(maskedIMGFinal, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) 
        averageLineMarkings = avgIntercept(frame, createLineMarkings) 
        lineImage = displayMarkings(frame, averageLineMarkings) 
        mergedImage = cv2.addWeighted(frame, 0.8, lineImage, 1, 1) 

        cv2.imshow('Display Test', mergedImage) # Display video
        # Setting up specific key to end video capture
        if cv2.waitKey(1) == ord('q'): # Set timer to 1 millisecond to seamlessly play video
            break
    capture.release()
    cv2.destroyAllWindows() # To remove all memory caches of the capture.

if __name__ == "__main__":
    imgSRC = '/Users/m20mi/Documents/Work/RT Lanes/test_image.jpg'
    vidSRC = '/Users/m20mi/Documents/Work/RT Lanes/test2.mp4'

    testVID = videoProcessor(vidSRC)

    # userInput = input("1 or 2")
    # if int(userInput) == 1:
    #     testVID = videoProcessor(vidSRC)
    # if int(userInput) == 2:
    #     testIMG = imageProcessor(imgSRC)
    #     cv2.imshow('Display Test', mergeResult(testIMG)) # Display image
    #     cv2.waitKey(0) # Displays image until <any> key is pressed.
    #     #plt.imshow(testIMG)
    #     #plt.show()



