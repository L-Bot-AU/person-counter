"""
TODO:
cv2.BackgroundSubtractor
"""

########## imports ##########
import imutils, cv2, itertools
import numpy as np
import random
from utilities import Handler, draw_str, distance, avgPoint
from conn_server import StubConnServer as ConnServer
#from conn_server import ConnServer


##########  magic values - senior library ##########
# [0..width*height]     minimum no. pixel of a person - affected by camera height
MIN_AREA = 4000
# [1..inf)              number of erode iterations
ERODE = 7
# [1..inf)              number of dilations iterations
DILATE = 30
# [0..255)              affected by lighting / shaddows
SENSITIVITY1 = 30
SENSITIVITY2 = 30
# [0..width+height)     max distance traversed by centroid - affected by camera height
MAX_DIST_PER_FRAME = 300
# [0..width-R]          left bound for process image crop
L = 0
# [L..width]            right bound of process image crop
R = 0


########## consts ##########
INF = float("inf")
# {True, False}         whether to display frame + annotations of centroids, traces
DEBUG = True
# [1..inf)              max number of people in frame
MAX_CENTROIDS = 2
# [0..inf)              minimum points required for count to happen
CONFIDENCE = 3
# [0..1]                threshold position (normalised distance to top)
CENTER = 0.65
# [0..1]                width of border
THRESHOLD_WIDTH = 0

# any substring of camera name, check names using `FocusUF.exe --list-cameras`
CAMERA_NAME, CAMERA_PORT = "c922 Pro Stream Webcam", 1
#CAMERA_NAME, CAMERA_PORT = "Microsoft Camera Front", 0
#CAMERA_NAME, CAMERA_PORT = "Microsoft Camera Rear", 1
def genColour(seed):
    random.seed(seed)
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )

########## processing frame ##########
class Main(Handler):
    """\
Contains various functions for processing image to detect library enters/exits
self.processFrame should be called externally and provided each frame from camera / video stream
updates database from self.server"""
    
    def __init__(self, cap, CAMERA_NAME, isStepped=False):
        super().__init__(cap, CAMERA_NAME, isStepped=isStepped)
        
        # compute thresholds for counting enters/exits
        self.referenceFrame = self.read()
        
        # for debugging to seek to a particular time in video
        #cap.set(cv2.CAP_PROP_POS_MSEC, (60*2+24)*1000)
        
        self.height, self.width, _ = self.referenceFrame.shape
        self.upThresh = int(self.height*(CENTER - THRESHOLD_WIDTH))
        self.downThresh = int(self.height*(CENTER + THRESHOLD_WIDTH))
        
        # first frame
        self.referenceFrame = self.prepareFrame(self.referenceFrame)
        
        # initialise variables
        self.prevCentroids = []
        self.trails = []
        self.server = ConnServer()

    def prepareFrame(self, rawFrame):
        return cv2.GaussianBlur(rawFrame, (5, 5), 0)[:, L:self.width-R]
        
    def getCentroids(self, thresh):
        """\
given a BW image, find centroids of possible people"""
        
        centroids = []
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in imutils.grab_contours(contours):
            # if the contour is large enough (this is probably a person)
            if cv2.contourArea(c) >= MIN_AREA:
                #print("Centroid area:", cv2.contourArea(c))
                
                # compute the bounding box for the contour, draw it on the frame, and update the text
                x, y, w, h = cv2.boundingRect(c)
                centroids.append((cv2.contourArea(c), (x+w//2, y+h//2)))
                
        centroids = [i[1] for i in sorted(centroids, reverse=True)][:MAX_CENTROIDS]
        
        # if two centroids is too close, they are the left and right side of a person's shoulders (head not detected)
        # thus, consider as 1 centroid (take average of points)
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                if distance(centroids[i], centroids[j]) < MAX_DIST_PER_FRAME:
                    centroids[i] = avgPoint(centroids[i], centroids[j])
                    del centroids[j]
        return centroids

    def track(self, prev, new):
        """\
find pairing of the sets of points (prev_i, new_i) which satify:
 - prev_i in prev
 - new_i in new
 - all euclidean distances <= MAX_DIST_PER_FRAME
 - sum of euclidean distances is minimised"""
        #print(prev, new)
        # highly inefficent, but like... did anyone ask?
        best = INF
        bestI = []
        for combPrev in itertools.permutations(prev):
            for combNew in itertools.permutations(new):
                cost = 0
                curr = []
                for p1, p2 in zip(combPrev, combNew):
                    d = distance(p1, p2)
                    if d <= MAX_DIST_PER_FRAME:
                        curr.append((p1, p2))
                        cost += d
                        cost += MAX_DIST_PER_FRAME*2 * abs(prev.index(p1) - new.index(p2))

                if not curr: continue
                # found a better pairing, set as the best pairing
                if len(curr) > len(bestI) or cost <= best:
                    best = cost
                    bestI = curr
        
        return bestI
    
    def getTrails(self, links):
        """\
updates self.trails from links"""
        newTrails = []
        for point1, point2 in links:
            for i in range(len(self.trails)):
                if point1 == self.trails[i][-1]:
                    newTrails.append(self.trails[i] + [point2])
                    break
            else:
                if point1[1] > self.downThresh or point1[1] < self.upThresh:
                    newTrails.append([point1, point2])
        self.trails = newTrails
        
    def processFrame(self, display):
        """\
given the frame, count poeple entering `server.add` and exiting `server.sub`"""
        
        # grab the current frame
        frameDelta = cv2.subtract(
            self.prepareFrame(display),
            self.referenceFrame
        )
        
        # remove background by filtering low values
        thresh = cv2.threshold(frameDelta, SENSITIVITY1, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        
        # erode, dilate the thresholded image to smooth / fill in gaps
        thresh = cv2.erode(thresh, None, iterations=ERODE)
        thresh = cv2.dilate(thresh, None, iterations=DILATE)
        thresh = cv2.threshold(thresh, SENSITIVITY2, 255, cv2.THRESH_BINARY)[1]
        
        # get list of people as pixel coordinates
        centroids = self.getCentroids(thresh)
        
        # for each centroid, find the previous coordinate it was at
        links = self.track(self.prevCentroids, centroids)
        #print(self.prevCentroids, centroids, links)
        self.getTrails(links)
        self.prevCentroids = centroids
        
        # detect people crossing threshold
        for trail in self.trails:
            # if trail is long enough to be considered more than "noise"
            if len(trail) >= CONFIDENCE:
                point1, point2 = trail[-2], trail[-1]
                # centroid of person went past the upper threshold (exiting library)
                if point1[1] > self.upThresh >= point2[1]: self.server.sub(1)
                
                # centroid of person went past the lower threshold (entering library)
                if point1[1] < self.downThresh <= point2[1]: self.server.add(1)
        
        if DEBUG:
            # draw tihe threshold line for entering or exiting the library
            cv2.line(display, (0, self.upThresh), (self.width, self.upThresh), (255, 0, 0))
            cv2.line(display, (0, self.downThresh), (self.width, self.downThresh), (255, 0, 0))
            
            # trace the path of all people
            for trails in self.trails:
                for i in range(len(trails)-1):
                    point1, point2 = trails[i], trails[i+1]
                    cv2.circle(display, point1, 1, (255, 0, 0), 5)
                    cv2.line(display, point1, point2, genColour(trails[0]), 3)

            #frameDelta[np.all(frameDelta == (0, 0, 0), axis=-1)] = (0, 255, 0)
            self.imshow("frameDelta", frameDelta)
            self.imshow("Thresh", thresh)
            
        return display

if __name__ == "__main__":
    # use either video stream of camera stream for image input
    # 1 works
    #video = "real_test_4/3.mp4"
    video = "video013.mp4"
    """#cap = cv2.VideoCapture(video)
    print("initialising camera")
    cap = cv2.VideoCapture(CAMERA_PORT)
    print("config 1")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    print("config 2")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)"""
    
    cap = cv2.VideoCapture(video)
    m = Main(cap, CAMERA_NAME, isStepped=0)
    m.start()

