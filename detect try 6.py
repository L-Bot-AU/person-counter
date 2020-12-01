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
# [0..width*height]     minimum no. pixel of a person - affected by camera height
MIN_AREA = 6000
# [0..255)              affected by lighting / shaddows
SENSITIVITY = 55
# [0..height//2)        width of border
THRESHOLD_WIDTH = 10
# [0..1]                threshold position (normalised distance to top)
CENTER = 0.5 #
# [0..width+height)     max distance traversed by centroid - affected by camera height
MIN_DIST_PER_FRAME = 0
# [0..width+height)     max distance traversed by centroid - affected by camera height
MAX_DIST_PER_FRAME = 300
# [1..inf)              number of dilations iterations
SMOOTHNESS = 32
# {True, False}         whether to display frame + annotations of centroids, traces
DEBUG = True

# magic values JNR
"""# [0..width*height]     minimum no. pixel of a person - affected by camera height
MIN_AREA = 15000
# [0..255)              affected by lighting / shaddows
SENSITIVITY = 60
# [0..height//2)        width of border
THRESHOLD_WIDTH = 10
# [0..1]                threshold position (normalised distance to top)
CENTER = 0.5 #
# [0..width+height)     max distance traversed by centroid - affected by camera height
MIN_DIST_PER_FRAME = 10
# [0..width+height)     max distance traversed by centroid - affected by camera height
MAX_DIST_PER_FRAME = 300
# [1..inf)              number of dilations iterations
SMOOTHNESS = 28
# {True, False}         whether to display frame + annotations of centroids, traces
DEBUG = True"""

# any substring of string name. check names using FocusUF-master\win32\focusuf\FocusUF.exe --list-cameras
CAMERA_NAME = "IZONE"
INF = float("inf")

class Main(Handler):
    def __init__(self, cap, isStepped=False):
        super().__init__(cap, CAMERA_NAME, isThreaded=False, isStepped=isStepped)
        # first frame
        _, self.initThresh = self.getFrame(self.cap.read()[1], doProcess=False)
        self.prevThresh = self.initThresh
        
        # compute thresholds for counting enters/exits
        self.height, self.width = self.initThresh.shape
        self.upThresh = int(self.height * CENTER - THRESHOLD_WIDTH)
        self.downThresh = int(self.height * CENTER + THRESHOLD_WIDTH)
        
        # initialise vars
        self.prevCentroids = []
        self.traceDebug = []
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
            frameDelta = cv2.absdiff(frameBlurred, self.initThresh)
            
            # remove background by filtering low values
            thresh = cv2.threshold(frameDelta, SENSITIVITY, 255, cv2.THRESH_BINARY)[1]
            
            # dilate the thresholded image to smooth / fill in gaps
            thresh = cv2.dilate(thresh, None, iterations=SMOOTHNESS)
        
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
    
    def opticalFlow(self, prevFrame, currFrame, prevPoints):
        npPoints = np.ndarray([len(prevPoints), 1, 2], dtype=np.float32)
        for i in range(len(prevPoints)):
            npPoints[i][0][0], npPoints[i][0][1] = prevPoints[i]
        
        currPoints, st, err = cv2.calcOpticalFlowPyrLK(
            prevFrame,
            currFrame,
            npPoints,
            None,
            winSize=(15,15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        #print(p1, st==1, err)
        if currPoints is None:
            return []
        currPoints = [tuple(map(round, val[0])) for val in currPoints.tolist()]
        return [val for val in zip(prevPoints, currPoints) if val[0] != val[1]]
    
    def processFrame(self, frame):
        """given the image capture object, count poeple entering (server.add) and exiting (server.sub) until end of stream"""
        # grab the current frame
        display, thresh = self.getFrame(frame)
        
        # get list of people as pixel coordinates
        self.prevCentroids.extend(self.getCentroids(thresh))
        # for each centroid, find the previous coordinate it was at
        #links = self.track(self.prevCentroids, centroids)
        
        links = self.opticalFlow(self.prevThresh, thresh, self.prevCentroids)
        self.prevCentroids = [i[1] for i in links]
        self.prevThresh = thresh
        
        # detect people crossing threshold
        for point1, point2 in links:
            # centroid of person went past the upper threshold (entering library)
            if point1[1] > self.upThresh >= point2[1]: self.server.add(1)
                
            # centroid of person went past the lower threshold (exiting library)
            if point1[1] < self.downThresh <= point2[1]: self.server.sub(1)
        
        if DEBUG:
            # draw the threshold line for entering or exiting the library
            cv2.line(display, (0, self.upThresh), (self.width, self.upThresh), (255, 0, 0))
            cv2.line(display, (0, self.downThresh), (self.width, self.downThresh), (255, 0, 0))
            
            # trace the path of all people
            self.traceDebug.extend(links)
            for point1, point2 in links: #self.traceDebug:
                cv2.circle(display, point1, 1, (255, 0, 0), 1)
                cv2.line(display, point1, point2, (0, 0, 255), 1)
                
            cv2.imshow("Thresh", imutils.resize(thresh, width=600))
            
        return display

if __name__ == "__main__":
    # use either video stream of camera stream for image input
    #video = "concept_test/multiple_1.mp4"
    video = "real_test_1/two_2.mp4"
    cap = cv2.VideoCapture(video)
    #cap = cv2.VideoCapture(2)
    
    m = Main(cap, isStepped=1)
    m.start()
    
