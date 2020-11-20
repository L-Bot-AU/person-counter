"""
TODO
https://github.com/opencv/opencv/blob/master/samples/python/video_threaded.py
make magic values more accessible
"""

from imutils.video import VideoStream
import imutils
import time
import cv2
import math
import itertools

from conn_server import StubConnServer as ConnServer
#from conn_server import ConnServer
server = ConnServer()

# magic values
MIN_AREA = 3500 # 0 <= MIN_AREA <= width*height                     minimum pixel area to be considered as a person - affected by camera height
SENSITIVITY = 55 # 0 <= SENSITIVITY < 255                           affected by lighting / shaddows
THRESHOLD_WIDTH = 8 # 0 <= THRESHOLD_WIDTH < height//2              width of border

FRAME_ID = 0
DEBUG = True
INF = float("inf")

def distance(p0, p1):
    """euclidian distanced between 2 coordinates"""
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def getFrame(cap):
    """given the image capture object, return (pretty debugging frame, processed frame)"""
    globals()["FRAME_ID"] += 1
    display = cap.read()
    if type(cap) == cv2.VideoCapture:
        display = display[1]
    if display is None:
        # cannot grab frame, end of stream
        return None, None
    
    # resize the frame (less information reduces computation)
    display = imutils.resize(display, width=300)
    # convert to grayscale (less information reduces computation)
    frame = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
    # blur it (remove noise)
    frame = cv2.GaussianBlur(frame, (21, 21), 0)
    
    return display, frame

def getCentroids(thresh):
    """given a BW image, find centroids of possible people"""
    centroids = []
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in imutils.grab_contours(contours):
        # if the contour is large enough (this is probably a person)
        if cv2.contourArea(c) >= MIN_AREA:
            # compute the bounding box for the contour, draw it on the frame, and update the text
            x, y, w, h = cv2.boundingRect(c)
            centroids.append((x+w//2, y+h//2))
    return centroids

def track(prev, new):
    """find pairing of the sets of points (prev, new) to minimise sum of euclidian distances"""
    best = INF
    bestI = []
    # highly inefficent, but like
    for combPrev in itertools.permutations(prev):
        for combNew in itertools.permutations(new):
            curr = sum(distance(p1, p2) for p1, p2 in zip(combPrev, combNew))
            # found a better pairing, set as the best pairing
            if curr < best:
                best = curr
                bestI = zip(combPrev, combNew)
    return list(bestI)

def main(cap):
    """given the image capture object, count poeple entering (server.add) and exiting (server.sub) until end of stream"""
    # initialise
    _, referenceFrame = getFrame(cap)
    numUp, numDown = 0, 0
    prevCentroids = []
    links = []
    traceDebug = []
    
    # loop over the frames of the video
    while True:
        # grab the current frame
        display, frame = getFrame(cap)
        # end of stream
        if frame is None: break
        
        # subtract background: absolute difference between the current frame and reference frame
        frameDelta = cv2.absdiff(frame, referenceFrame)
        # remove background by filtering low values
        thresh = cv2.threshold(frameDelta, SENSITIVITY, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to smooth / fill in gaps
        thresh = cv2.dilate(thresh, None, iterations=3)

        # get list of people as pixel coordinates
        centroids = getCentroids(thresh)
        # for each centroid, find the previous coordinate it was at
        links = track(prevCentroids, centroids)
        
        # compute thresholds for counting enters/exits
        height, width = frame.shape
        upThresh = int(height//2 - THRESHOLD_WIDTH)
        downThresh = int(height//2 + THRESHOLD_WIDTH)
        
        # detect people crossing threshold
        for point1, point2 in links:
            # centroid of person went past the upper threshold (entering library)
            if point1[1] > upThresh >= point2[1]:
                server.add(1)
                numUp += 1
                
            # centroid of person went past the lower threshold (exiting library)
            if point1[1] < downThresh <= point2[1]:
                server.sub(1)
                numDown += 1
        
        if DEBUG:
            # draw the threshold line for entering or exiting the library
            cv2.line(display, (0, upThresh), (width, upThresh), (255, 0, 0))
            cv2.line(display, (0, downThresh), (width, downThresh), (255, 0, 0))
            
            # draw the counter values on the screen
            cv2.putText(display, f"{numUp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display, f"{numDown}", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # trace the path of all people
            traceDebug.extend(links)
            for point1, point2 in traceDebug:
                cv2.circle(display, point1, 1, (0, 0, 255), 1)
                cv2.line(display, point1, point2, (0, 255, 0), 1)
                
            # show the frame
            cv2.imshow("Frame", imutils.resize(display, width=800))
            cv2.imshow("Thresholded frame", thresh)
            
            # if the `q` key is pressed, break from the loop
            if cv2.waitKey(1) & 0xFF == ord("q"): break
            #cv2.waitKey(0)
            
        prevCentroids = centroids
        
    # cleanup the camera, close open windows if in debug mode
    if type(cap) == cv2.VideoCapture:
        cap.release()
    else:
        cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # use either video stream of camera stream for image input
    video = "concept_test/jiggle_1.mp4"
    cap = cv2.VideoCapture(video)
    #cap = VideoStream(src=0).start()
    
    main(cap)
