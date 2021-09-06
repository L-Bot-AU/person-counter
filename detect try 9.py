########## imports ##########
from utilities import distance, avgPoint, genColour, Handler
import itertools
import imutils
import cv2


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

########## consts ##########
INF = float("inf")
# {True, False}         whether to display frame + annotations of centroids, traces
DEBUG = True
# [1..inf)              max number of people in frame
MAX_CENTROIDS = 3
# [0..inf)              minimum points required for count to happen
CONFIDENCE = 3
# [0..1]                threshold position (normalised distance to top)
CENTER = 0.45
# [0..1]                width of border
THRESHOLD_WIDTH = 0

# any substring of camera name, check names using `FocusUF.exe --list-cameras`
CAMERA_NAME, CAMERA_PORT = "c922 Pro Stream Webcam", 1
#CAMERA_NAME, CAMERA_PORT = "Microsoft Camera Front", 0
#CAMERA_NAME, CAMERA_PORT = "Microsoft Camera Rear", 1


########## processing frame ##########
class Main(Handler):
    """\
contains various functions for processing image to detect library enters/exits
self.processFrame is called by super() and provided each frame from the camera / video stream
updates database from self.server"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def init(self, referenceFrame):
        """\
:param referenceFrame: the first frame in the video capture, should only contain backgroud
:return: None"""
        
        # compute thresholds for counting enters/exits
        self.referenceFrame = referenceFrame
        self.height, self.width, _ = referenceFrame.shape
        self.upThresh = int(self.height*(CENTER - THRESHOLD_WIDTH))
        self.downThresh = int(self.height*(CENTER + THRESHOLD_WIDTH))
        
        # initialise variables
        self.prevCentroids = []
        self.trails = []
    
    def prepareFrame(self, rawFrame):
        """\
:param rawFrame: image
:return: image after the required transformations, only this image is used and processed"""
        return cv2.GaussianBlur(rawFrame, (5, 5), 0)
    
    def getCentroids(self, thresh):
        """\
:param thresh: BW image
:return: the centroids of all countours which have area >= MIN_AREA, sorted by decreasing area. close centroids are merged together"""
        
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
                if max(i, j) < len(centroids) and distance(centroids[i], centroids[j]) < MAX_DIST_PER_FRAME:
                    centroids[i] = avgPoint(centroids[i], centroids[j])
                    del centroids[j]
        return centroids
    
    def track(self, prev, new):
        """\
:param prev: list of points representing centroids of the previous frame, sorting by decreasing area
:param new: list of points representing centroids of the current frame, sorting by decreasing area
:return: list of optimal pairs of points from prev to new such that:
 - all euclidean distances <= MAX_DIST_PER_FRAME
 - sum of euclidean distances is minimised
 - order of points is as similar as possible to its input order"""
        #print(prev, new)
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
    
    def updateTrails(self, links):
        """\
:param links: list of pairs of points
:return: None

extends self.trails from links, or remove the trail if it doesn't exist anymore"""
        newTrails = []
        for point1, point2 in links:
            for i in range(len(self.trails)):
                if point1 == self.trails[i][-1]:
                    newTrails.append(self.trails[i] + [point2])
                    break
            else:
                # trail can only start outside boundary range
                if point1[1] > self.downThresh or point1[1] < self.upThresh:
                    newTrails.append([point1, point2])
        self.trails = newTrails
        
    def sendCount(self):
        """\
:param self.trails: list of list of points
:return: None

uses self.trails to detect when a centroid has just crossed the boundary
depending on its direction, sends an add(1) or sub(1) to the server"""
        
        for trail in self.trails:
            # if trail is long enough to be considered more than "noise"
            if len(trail) >= CONFIDENCE:
                point1, point2 = trail[-2], trail[-1]
                # centroid of person went past the upper threshold (exiting library)
                if point1[1] > self.upThresh >= point2[1]: self.server.sub(1)
                
                # centroid of person went past the lower threshold (entering library)
                if point1[1] < self.downThresh <= point2[1]: self.server.add(1)
    
    def processFrame(self, display):
        """\
:param display: image
:return: display image for debugging screen

count poeple entering and exiting"""
        
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
        self.prevCentroids = centroids
        self.updateTrails(links)
        
        # detect people crossing threshold, and send to server
        self.sendCount()
        
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

            #frameDelta[np.all(frameDelta == (0, 0, 0), axis=-1)] = (0, 255, 0) #greeeen
            self.imshow("frameDelta", frameDelta)
            self.imshow("Thresh", thresh)
            
        return display

if __name__ == "__main__":
    # source = "real_test_4/3.mp4"
    source = "real_test_5/video013.mp4"
    # source = 1 # front webcam
    m = Main(source, CAMERA_NAME, debug=True, seekTime=0)
    m.start()

