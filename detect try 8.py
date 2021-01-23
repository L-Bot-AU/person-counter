"""
TODO
experiment with: filter out blue
make magic values more accessible (how?)
"""

import imutils
import time
import cv2
import math
import itertools
import numpy as np

from utilities import Handler, draw_str, distance
from conn_server import StubConnServer as ConnServer
#from conn_server import ConnServer

# magic values SNR
# [0..height//2)        width of border
THRESHOLD_WIDTH = 10
# [0..1]                threshold position (normalised distance to top)
CENTER = 0.6 #
# [0..width*height]     minimum no. pixel of a person - affected by camera height
MIN_AREA = 8000
# [0..255)              affected by lighting / shaddows
SENSITIVITY = 15
# [0..width+height)     max distance traversed by centroid - affected by camera height
MIN_DIST_PER_FRAME = 0
# [0..width+height)     max distance traversed by centroid - affected by camera height
MAX_DIST_PER_FRAME = 300
# [1..inf)              number of erode iterations
ERODE = 5
# [1..inf)              number of dilations iterations
DILATE = 21
# [0..inf)              minimum points required for count to happen
CONFIDENCE = 3
# {True, False}         whether to display frame + annotations of centroids, traces
DEBUG = True

# any substring of string name. check names using FocusUF-master\win32\focusuf\FocusUF.exe --list-cameras
CAMERA_NAME = "IZONE"

# consts
INF = float("inf")

class Main(Handler):
    def __init__(self, cap, isStepped=False):
        super().__init__(cap, CAMERA_NAME, isThreaded=False, isStepped=isStepped)
        # first frame
        _, self.referenceFrame = self.getFrame(self.cap.read()[1], doProcess=False)
        self.prevThresh = self.referenceFrame
        
        # compute thresholds for counting enters/exits
        self.height, self.width = self.referenceFrame.shape
        self.upThresh = int(self.height*CENTER - THRESHOLD_WIDTH)
        self.downThresh = int(self.height*CENTER + THRESHOLD_WIDTH)
        
        # initialise vars
        self.prevCentroids = []
        self.trails = []
        self.server = ConnServer()
        
    def getFrame(self, display, doProcess=True):
        """given the initial frame, return (pretty debugging frame, processed frame)"""
        # resize the frame (less information reduces computation)
        display = imutils.resize(display, width=600)
        
        # only green channel (more contrast against blue carpet)
        display[:, :, 0] = 0
        display[:, :, 2] = 0
        
        # convert to grayscale (less information reduces computation)
        frameBW = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
        
        # blur it (remove noise)
        frameBlurred = cv2.GaussianBlur(frameBW, (21, 21), 0)
        
        if doProcess:
            # subtract background: absolute difference between the current frame and reference frame
            frameDelta = cv2.absdiff(frameBlurred, self.referenceFrame)
            
            # remove background by filtering low values
            thresh = cv2.threshold(frameDelta, SENSITIVITY, 255, cv2.THRESH_BINARY)[1]
            
            # dilate the thresholded image to smooth / fill in gaps
            thresh = cv2.erode(thresh, None, iterations=ERODE)
            thresh = cv2.dilate(thresh, None, iterations=DILATE)
        
            return display, thresh
        
        else:
            return display, frameBlurred

    def getCentroids(self, thresh):
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

    def track(self, prev, new):
        """find pairing of the sets of points (prev, new) to minimise sum of euclidian distances"""
        best = INF
        bestI = []
        # highly inefficent, but like
        for combPrev in itertools.permutations(prev):
            for combNew in itertools.permutations(new):
                d = [distance(p1, p2) for p1, p2 in zip(combPrev, combNew)]
                # found a better pairing, set as the best pairing
                if all(MIN_DIST_PER_FRAME <= val <= MAX_DIST_PER_FRAME for val in d) and sum(d) <= best:
                    best = sum(d)
                    bestI = zip(combPrev, combNew)
        return list(bestI)
    
    def getTrails(self, links):
        """updates self.trails"""
        newTrails = []
        for point1, point2 in links:
            for i in range(len(self.trails)):
                if point1 == self.trails[i][-1]:
                    newTrails.append(self.trails[i] + [point2])
                    break
            else:
                newTrails.append([point1, point2])
        self.trails = newTrails
        
    def processFrame(self, frame):
        """given the image capture object, count poeple entering (server.add) and exiting (server.sub) until end of stream"""
        # grab the current frame
        display, thresh = self.getFrame(frame)
        
        # get list of people as pixel coordinates
        centroids = self.getCentroids(thresh)
        
        # for each centroid, find the previous coordinate it was at
        links = self.track(self.prevCentroids, centroids)
        
        # join self.trails to links
        self.getTrails(links)
        
        self.prevCentroids = centroids
        self.prevThresh = thresh
        
        # detect people crossing threshold
        for trail in self.trails:
            if len(trail) >= CONFIDENCE:
                point1, point2 = trail[-2], trail[-1]
                # centroid of person went past the upper threshold (entering library)
                if point1[1] > self.upThresh >= point2[1]: self.server.add(1)
                
                # centroid of person went past the lower threshold (exiting library)
                if point1[1] < self.downThresh <= point2[1]: self.server.sub(1)
        
        if DEBUG:
            # draw the threshold line for entering or exiting the library
            cv2.line(display, (0, self.upThresh), (self.width, self.upThresh), (255, 0, 0))
            cv2.line(display, (0, self.downThresh), (self.width, self.downThresh), (255, 0, 0))
            
            # trace the path of all people
            for trails in self.trails:
                for i in range(len(trails)-1):
                    point1, point2 = trails[i], trails[i+1]
                    cv2.circle(display, point1, 1, (255, 0, 0), 1)
                    cv2.line(display, point1, point2, (0, 0, 255), 1)
            
            cv2.imshow("Thresh", imutils.resize(thresh, width=600))
            
        return display

if __name__ == "__main__":
    # use either video stream of camera stream for image input
    video = "real_test_1/two_1.mp4"
    #video = "real_test_2/jacket_phone_1.mp4"
    #cap = cv2.VideoCapture(video)
    cap = cv2.VideoCapture(2)
    
    m = Main(cap, isStepped=0)
    m.start()
    
