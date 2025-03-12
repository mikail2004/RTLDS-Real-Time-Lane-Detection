# Mikail Usman • 2/6/2025
# Real Time Lane Detection System (RTLDS)

from flask import Flask, render_template, request, Response
import lanes as LD
import cv2
import os

# Setting up Flask and required Paths
app = Flask(__name__, template_folder='web', static_folder='static')
uploadsFolderPath = r'/Users/m20mi/Documents/Work/RT Lanes/static/uploads'
holdFolderPath = r'/Users/m20mi/Documents/Work/RT Lanes/static/hold'
app.config['UPLOAD_FOLDER'] = uploadsFolderPath # Directory for uploaded images
app.config['PROCESSED_FOLDER'] = holdFolderPath # Directory for processed images

# Home page (With default value for HTML JINJA img src)
@app.route("/")
def home():
    return render_template("index.html", imageURL="", videoURL="")

# Route accessed if user uploads image (@"/success" must correspond with HTML form <action> value)
@app.route("/imageUpload", methods=['POST'])
def successImage():
    # If method from HTML form is 'POST':
    if request.method == 'POST':
        # Saving the uploaded image to the '/uploads' folder
        f = request.files['file']
        filePathOriginal = os.path.join(app.config['UPLOAD_FOLDER'], f.filename) # Used to dynamically concat file paths 
        f.save(filePathOriginal)

        # Processing image with lanes.py, using <cv2.imwrite(path, img)> to save to '/hold' folder
        imageLD = LD.imageProcessor(filePathOriginal)
        filePathProcessed = os.path.join(app.config['PROCESSED_FOLDER'], 'savedImage.jpg')
        cv2.imwrite(filePathProcessed, imageLD)

        return render_template("index.html", imageURL="savedImage.jpg", videoURL="")

# Route (to page) accessed if user uploads video
@app.route("/videoUpload", methods=['POST'])
def successVideo():
    # If method from HTML form is 'POST':
    if request.method == 'POST':
        f = request.files['file']
        filePathOriginal = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(filePathOriginal)

        # Setting path to new video (with processed frames)
        filePathProcessed = os.path.join(app.config['PROCESSED_FOLDER'], 'savedVideo.mp4')

        # Generating new video (with processed frames)
        LD.videoProcessorWeb(filePathOriginal, filePathProcessed)

        return render_template("index.html", imageURL="", videoURL="savedVideo.mp4")
    
# Response (Without re-route to page) for live camera feed
@app.route("/liveFeed")
def successLive():
    # For the purpose of having a stream where each part replaces the previous part,
    # the 'multipart/x-mixed-replace' content type must be used.
    # Streams a sequence of independent JPEG pictures (Motion JPEG) -> single client only outside of debug.
    return Response(LD.processFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Deletes contents of the 'uploads' and 'hold' folders.
@app.route("/clearData", methods=['POST'])
def clearData():
    uploadFiles = os.listdir(uploadsFolderPath)
    holdFiles = os.listdir(holdFolderPath)

    # Delete all files within a directory
    for file in uploadFiles:
        filePath = os.path.join(uploadsFolderPath, file)
        if os.path.isfile(filePath):
            os.remove(filePath)

    for file in holdFiles:
        filePath = os.path.join(holdFolderPath, file)
        if os.path.isfile(filePath):
            os.remove(filePath)
    
    return render_template('index.html', imageURL="", videoURL="")

if __name__ == "__main__":
    app.run(debug=True)