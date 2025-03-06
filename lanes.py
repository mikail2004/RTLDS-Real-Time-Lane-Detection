# Mikail Usman • 2/6/2025
# Real Time Lane Detection System (RTLDS)

import cv2 # From full package [opencv-contrib-python]
import numpy as np
import matplotlib.pyplot as plt

# -- Creating a Mask Outlining Region of Interest -- 
def polygonMask(image):
    height = image.shape[0] # Getting height (no.of rows) using shape of the image array.
    # Creating a polygon to cover the region of interest (Lane):
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)] # Uses coordinates of lane zone from image (courtesy of pyplot).
    ])
    # Creates black mask over image - by making an array of zeros:
    # || Creates black image with same dimensions as original image
    mask = np.zeros_like(image) # (Each value = pixel intensity of zero I.E Black) with the same no. of rows/cols as OG image.
    cv2.fillPoly(mask, polygons, 255) # Area bounded by the polygonal contour will be fully white (255).
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
        slope, intercept = parameters[0], parameters[1] # As slope is at the first index of the returned vector <parameters>

        # To see if the slope/intercept returned is of the right side of the lane or left.
        # Lines on the left will have a negative slope (m<0)
        # Lines on the right will have a positive slope (m>0)
        if slope < 0: 
            leftFit.append((slope, intercept))
        else:
            rightFit.append((slope, intercept))

    # Average out all intercepts and slopes of each side of the lane:
    leftFitAvg = np.average(leftFit, axis=0) if leftFit else None
    rightFitAvg = np.average(rightFit, axis=0) if rightFit else None

    # Combining intercepts and slopes with coordinates to create a proper line (y=mx+c)
    leftLine = calcCoords(image, leftFitAvg) if leftFitAvg is not None else None
    rightLine = calcCoords(image, rightFitAvg) if rightFitAvg is not None else None

    return np.array([line for line in [leftLine, rightLine] if line is not None])

# -- Calculate Coordinates for Optimization Averaging Function --
def calcCoords(image, lineParameters):
    # Making sure there are no errors if a lane on either side has not been detected
    if lineParameters is None or not isinstance(lineParameters, (list, tuple, np.ndarray)):
        return None  # Return None if no valid line parameters
    if isinstance(lineParameters, np.ndarray) and lineParameters.size == 2:
        slope, intercept = lineParameters
    else:
        return None 

    # Finding Coordinates needed to draw lane line using averaged out slope and intercept (y=mx+c)
    y1 = image.shape[0] # Fixing minimum height to 0 (starting point of the image height).
    y2 = int(y1*3/5) # Fixing maximum height (based on image height) to be about 3/5th the image height.
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

# -- Accessing and Applying Image Processing -- 
def imageProcessor(inputIMG):
    image = cv2.imread(inputIMG) # Reads image and returns it as a multi-dimensional numPy array containing pixel intensities.
    imageResize = cv2.resize(image, (1279, 704))
    laneImageCopy = np.copy(imageResize) # Copying image array || Keep original separate from the one we will process.
    imageGS = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # Converting image (laneImageCopy) to grayscale. 
    imageBlur = cv2.GaussianBlur(imageGS, (5,5), 0) # Gaussian blur to reduce noise w/ 5x5 Kernel and deviation of 0.
    imageCanny = cv2.Canny(imageBlur, 50, 150) # Executing Canny function with thresholds b/w 50 and 150.
    maskedIMGFinal = polygonMask(imageCanny) # Executing polygonMask() to outline region of interest.

    # Combining Generated Lane Lines and Original Image
    # Hough Transform via func(image, pixel precision for bin size, theta precision for bin size, threshold)
    createLineMarkings = cv2.HoughLinesP(maskedIMGFinal, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) 
    averageLineMarkings = avgIntercept(laneImageCopy, createLineMarkings) # Average out multiple markings into one for each lane side.
    lineImage = displayMarkings(laneImageCopy, averageLineMarkings) # Displaying generated lane markings onto OG media.
    mergedImage = cv2.addWeighted(laneImageCopy, 0.8, lineImage, 1, 1) # Take weighted sum of OG image array and lineImage array to find resultant merged image.
    return mergedImage

# -- Accessing and Applying Video Processing -- 
def videoProcessor(videoSRC):
    capture = cv2.VideoCapture(videoSRC) # Receives video from a link
    # If video capture is active, start loop
    while(capture.isOpened()): 
        # Ret is a boolean, frame is the from the video.
        # Ret indicates whether frame was succesfully read from the video file or not.
        ret, frame = capture.read() 

        # Breaks loop if no more frames (If video is over)
        if not ret or frame is None:
            break
        
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

# -- Video Processing for Web Clients (Flask) -- 
def videoProcessorWeb(videoSRC, savePath):
    capture = cv2.VideoCapture(videoSRC) 

    # Getting original video fps and frame height, width
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) # Tuple of 2

    # Creating a new video using CV2 and the processed frames
    # Use codec 'h.264' to be compatible with the web
    videoProcessed = cv2.VideoWriter(savePath, cv2.VideoWriter_fourcc(*'h264'), fps, size)

    while(capture.isOpened()): 
        ret, frame = capture.read() 

        if not ret or frame is None:
            break
        
        imageCanny = cv2.Canny(frame, 50, 150)
        maskedIMGFinal = polygonMask(imageCanny) 
        createLineMarkings = cv2.HoughLinesP(maskedIMGFinal, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) 
        averageLineMarkings = avgIntercept(frame, createLineMarkings) 
        lineImage = displayMarkings(frame, averageLineMarkings) 
        mergedImage = cv2.addWeighted(frame, 0.8, lineImage, 1, 1) 

        videoProcessed.write(mergedImage) # Writes each frame to the new video

    capture.release()
    videoProcessed.release() # Once all frames are added via loop, video is finalized and saved
    cv2.destroyAllWindows() # To remove all memory caches of the capture.

if __name__ == "__main__":
    # All input media must be 1279 x 704 pixels
    imgSRC = '/Users/m20mi/Documents/Work/RT Lanes/test1.jpg'
    vidSRC = '/Users/m20mi/Documents/Work/RT Lanes/testVideo.mp4'

    print("")
    print("RTLDS")
    userInput = input("Run Video <1> or Run Image <2> >>> ")
    if int(userInput) == 1:
        testVID = videoProcessor(vidSRC)
    if int(userInput) == 2:
        #plt.imshow(testIMG)
        #plt.show()

        cv2.imshow('Display Test', imageProcessor(imgSRC)) # Display image
        cv2.waitKey(0) # Displays image until <any> key is pressed.