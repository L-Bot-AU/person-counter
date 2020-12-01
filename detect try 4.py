"""
TODO
maybe cap change in speed / direction
make magic values more accessible (how?)
after receiving real world data, deal with shaddows (how?)
"""

from imutils.video import VideoStream
import imutils
import time
import cv2
import math
import itertools
import subprocess

from utilities import Handler, draw_str
from conn_server import StubConnServer as ConnServer
#from conn_server import ConnServer

# [0..width*height]     minimum no. pixel of a person - affected by camera height
MIN_AREA = 11000
# [0..255)              affected by lighting / shaddows
SENSITIVITY = 55 
# [0..height//2)        width of border
THRESHOLD_WIDTH = 60
# [0..width+height)     max distance traversed by centroid - affected by camera height
MAX_DIST_PER_FRAME = 150
# [1..inf)              number of dilations iterations
SMOOTHNESS = 28
# {True, False}         whether to display frame + annotations of centroids, traces
DEBUG = True
# any substring of string name. check names using FocusUF-master\win32\focusuf\FocusUF.exe --list-cameras
CAMERA_NAME = "IZONE"


def distance(p0, p1):
    """euclidian distanced between 2 coordinates"""
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def getFrame(display):
    """given the initial frame, return (pretty debugging frame, processed frame)"""
    # resize the frame (less information reduces computation)
    display = imutils.resize(display, width=600)
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
            centroids.append((cv2.contourArea(c), (x+w//2, y+h//2)))
    return [i[1] for i in sorted(centroids, reverse=True)][:2]

def track(p1, newPoints):
    """find pairing of the sets of points (prev, new) to minimise sum of euclidian distances"""
    # limit false detection - people wont move faster than this
    best = MAX_DIST_PER_FRAME
    bestI = None
    # highly inefficent, but like
    for p2 in newPoints:
        # found a better pairing, set as the best pairing
        if 0 < distance(p1, p2) < best:
            best = distance(p1, p2)
            bestI = p2
    return bestI

class Main(Handler):
    def __init__(self, cap, **kwargs):
        super().__init__(cap, CAMERA_NAME, **kwargs)
        # initialise
        _, self.referenceFrame = getFrame(self.cap.read()[1])
        self.numUp, self.numDown = 0, 0
        self.prevCentroid = None
        self.traceDebug = []
        self.server = ConnServer()
        
    def processFrame(self, frame):
        """given the image capture object, count poeple entering (server.add) and exiting (server.sub) until end of stream"""
        # grab the current frame
        display, frame = getFrame(frame)
        
        # compute thresholds for counting enters/exits
        height, width = frame.shape
        upThresh = int(height//2 - THRESHOLD_WIDTH)
        downThresh = int(height//2 + THRESHOLD_WIDTH)
        
        # subtract background: absolute difference between the current frame and reference frame
        frameDelta = cv2.absdiff(frame, self.referenceFrame)
        # remove background by filtering low values
        thresh = cv2.threshold(frameDelta, SENSITIVITY, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to smooth / fill in gaps
        thresh = cv2.dilate(thresh, None, iterations=SMOOTHNESS)
        
        # get list of people as pixel coordinates
        # for each centroid, find the previous coordinate it was at
        centroids = getCentroids(thresh)
        if self.prevCentroid:
            newCentroid = track(self.prevCentroid, centroids)
            if newCentroid:
                # centroid of person went past the upper threshold (entering library)
                if self.prevCentroid[1] > upThresh >= newCentroid[1]:
                    self.server.add(1)
                    self.numUp += 1
                    
                # centroid of person went past the lower threshold (exiting library)
                if self.prevCentroid[1] < downThresh <= newCentroid[1]:
                    self.server.sub(1)
                    self.numDown += 1
                    
                if DEBUG:
                    self.traceDebug.append((self.prevCentroid, newCentroid))
                    
            self.prevCentroid = newCentroid
        else:
            validStarts = [i for i in centroids if i[1] < downThresh or i[1] > upThresh]
            if validStarts:
                self.prevCentroid = validStarts[0]
            
        if DEBUG:
            cv2.imshow("Thresh", thresh)
            
            # draw the threshold line for entering or exiting the library
            cv2.line(display, (0, upThresh), (width, upThresh), (255, 0, 0))
            cv2.line(display, (0, downThresh), (width, downThresh), (255, 0, 0))
            
            # draw the counter values on the screen
            draw_str(display, f"{self.numUp}", (width - 25, 30))
            draw_str(display, f"{self.numDown}", (width - 25, height - 20))
            
            for point1, point2 in self.traceDebug:
                cv2.circle(display, point1, 1, (0, 0, 255), 1)
                cv2.line(display, point1, point2, (0, 255, 0), 1)
            for point in centroids:
                cv2.circle(display, point, 1, (0, 0, 255), 5)
            cv2.waitKey(0)
            
        return display

if __name__ == "__main__":
    # use either video stream of camera stream for image input
    video = "real_test/two_1.mp4"
    #video = "concept_test/fast_1.mp4"
    cap = cv2.VideoCapture(video)
    #cap = cv2.VideoCapture(2)
    
    m = Main(cap, isThreaded=False)
    m.start()
    
