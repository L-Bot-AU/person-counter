"""
TODO
make centroid tracker
https://github.com/opencv/opencv/blob/master/samples/python/video_threaded.py
non-zero width theshold
make magic values more accessible
"""

from imutils.video import VideoStream
from mosse.mosse import MOSSE as Tracker
#from centroid_track import CentroidTrack as Tracker
import imutils
import time
import cv2
import math

# magic values
MIN_AREA = 3000 # minimum pixel area of person - affected by camera height
THRESHOLD = 50 # sensitivity - affected by lighting / shaddows
DEBUG = True

def distance(p0, p1): #NOT USED
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def getFrame(cap):
    frame = cap.read()
    if type(cap) == cv2.VideoCapture:
        frame = frame[1]
    if frame is None:
        return None, None
    
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=200)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (21, 21), 0)
    return frame, grey

def main(cap):
    # initialise
    _, firstFrame = getFrame(cap)
    trackers = []
    numUp, numDown = 0, 0
    frameId = 0
    
    # loop over the frames of the video
    while True:
        # grab the current frame
        display, frame = getFrame(cap)
        
        # could not be grabbed, reached the end of the stream
        if frame is None: break
        
        # compute the absolute difference between the current frame and the reference frame
        frameDelta = cv2.absdiff(frame, firstFrame)
        # remove background by filtering low values
        thresh = cv2.threshold(frameDelta, THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to fill in holes, then find contours on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=3)
        
        # comment
        for tracker in trackers:
            prev = tuple(map(int, tracker.pos))
            tracker.update(frame)
            new = tuple(map(int, tracker.pos))
            if DEBUG:
                cv2.circle(display, prev, 10, (0, 0, 255), 10)
                cv2.circle(display, new, 10, (0, 255, 0), 10)

            # comment
            """if CONDITION:
                numUp += 1
            elif CONDITION:
                numDown += 1"""
                
        trackers = []
        
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in imutils.grab_contours(contours):
            # if the contour is large enough (this is probably a person)
            if cv2.contourArea(c) >= MIN_AREA:
                # compute the bounding box for the contour, draw it on the frame, and update the text
                x, y, w, h = cv2.boundingRect(c)
                trackers.append(Tracker(frame, (x, y, x+w, y+h)))
        
        if DEBUG:
            # draw the text and timestamp on the frame
            cv2.putText(display, f"{numUp}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)
            cv2.putText(display, f"{numDown}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)

            # draw the people
            for tracker in trackers:
                tracker.draw_state(frame)
                
            # show the frame
            cv2.imshow("Security Feed", imutils.resize(display, width=800))
            cv2.imshow("Thresh", thresh)
            #cv2.imshow("Frame Delta", frameDelta)
            
            # if the `q` key is pressed, break from the lop
            if cv2.waitKey(1) & 0xFF == ord("q"): break
            #if frameId % 3 == 0: cv2.waitKey(0)
            
        # comment
        frameId += 1
    
    # cleanup the camera and close any open windows
    if type(cap) == cv2.VideoCapture:
        cap.release()
    else:
        cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    video = "concept_test/normal_1.mp4"
    cap = cv2.VideoCapture(video)
    #cap = VideoStream(src=0).start()
    
    main(cap)
