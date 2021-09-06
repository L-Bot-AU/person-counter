########## imports ##########
from conn_server import StubConnServer as ConnServer
#from conn_server import ConnServer

import subprocess
import random
import imutils
import math
import cv2

########## constants ##########
# [0..width-R]          left bound for cropping image before being processed
L = 0
# [L..width]            right bound for cropping image before being processed
R = 0
# frames will be scaled to this resolution before being processed
PROCESSING_RES = 720
# frames will be scaled to this resolution before being displayed, should not be greater than PROCESSING_RES
DISPLAY_RES = 480

########## helper functions ##########
def distance(p0, p1):
    """\
:param p1: 2d coordinate as a tuple of integers
:param p2: 2d coordinate as a tuple of integers
:return: aeuclidian distanced between the 2 coordinates"""
    return math.sqrt((p0[0] - p1[0])*(p0[0] - p1[0]) + (p0[1] - p1[1])*(p0[1] - p1[1]))

def avgPoint(p0, p1):
    """\
:param p1: 2d coordinate as a tuple of integers
:param p2: 2d coordinate as a tuple of integers
:return: midpoint of p1 and p2"""
    return (p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2

def clock():
    """\
:return: timestamp of the current frame"""
    return cv2.getTickCount() / cv2.getTickFrequency()

def draw_str(frame, s : str, target : (int, int)):
    """\
:param frame: image
:param s: the string of text to write on the frame
:param target: location of the text
:return: None, the frame is changed in place"""
    x, y = target
    
    # shaddow
    cv2.putText(frame, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    # text
    cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

def runCmd(cmd : str):
    """\
:param cmd: command for the terminal
:return: None, but prints the output of the command"""
    print(subprocess.getoutput(cmd.split(" ")))

def genColour(seed : int):
    """\
:param seed: any integer
:return: rgb value, represented as a tuple of 3 integers

assigns a unique colour for every seed"""
    random.seed(seed)
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )

########## Handler ##########
class Handler:
    """\
wrapper for displaying and processing video
handles the high level processing
use: `p` to toggle stepping mode and playthrough mode, ` ` to step, `q` to quit"""
    
    def __init__(self, source, CAMERA_NAME=None, debug=False, seekTime=None):
        """\
:param cap: cv2.VideoCapture object
:param CAMERA_NAME: camera name e.g. "c922 Pro Stream Webcam"
:param isStepped: debugger flag indicating whether you progress through frames one at a time on keypress
:param seekTime: jump to a particular time in the video, given in seconds
:return: None

initialises capture, set focus mode to auto
"""
        try:
            self.init
        except AttributeError:
            raise NameError("requires a function self.init(referenceFrame) which processes the first frame of the video capture")
        try:
            self.processFrame
        except AttributeError:
            raise NameError("requires a function self.processFrame(frame) which uses self.server to update the count, and return displayFrame")
        
        # initialise capture
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROCESSING_RES)
        if seekTime:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, seekTime*1000)
            
        # set focus mode to auto
        if not CAMERA_NAME:
            runCmd("focusuf\\FocusUF.exe --list-cameras")
            raise Exception("select a camera")
        runCmd(f"focusuf\\FocusUF.exe --camera-name {CAMERA_NAME} --focus-mode-manual --exposure-mode-manual")
        
        # set debugging variables
        self.debug = debug
        self.isStepped = False
        self.stopped = False
                    
        # initialise connection to server, to send a count as people enter/exit
        self.server = ConnServer()

        # initialise processes for subroutines
        self.init(self.read())
        
    def togglePause(self, *args, **kwargs):
        print(args, kwargs)
        
    def read(self):
        """\
:return: the most recent image captured from self.cap, scaled to PROCESSING_RES"""
        
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        frame = imutils.resize(frame, height=PROCESSING_RES)
        return frame[:, L:frame.shape[1]-R]
    
    def imshow(self, windowName, frame):
        """\
:param windowName: the title to displayed above the frame
:param frame: the frame to be scaled to DISPLAY_RES and displayed
:return: None"""
        
        cv2.imshow(windowName, imutils.resize(frame, height=DISPLAY_RES))
    
    def pop(self, frame, timeElapsed):
        """\
:param frame: frame to be display after being processed
:param timeElapsed: time taken for entire processing, given in seconds
:return: None

Deal with frame after self.processFrame"""
        
        if self.debug:
            draw_str(frame, f"latency: {round(timeElapsed * 1000, 3)} ms", (20, 20))
            draw_str(frame, f"{round(1 / timeElapsed, 3)} frames/s", (20, 40))
            self.imshow("debug", frame)
        
        if self.isStepped:
            k = cv2.waitKey(0) # waits infinitely for a keypress
        else:
            k = cv2.waitKey(1) # wait 1ms for a keypress
            
        if k == ord("q"):
            self.stop()
        elif k == ord("p"):
            self.isStepped = not self.isStepped
        elif not self.isStepped and k == ord(" "):
            k = cv2.waitKey(0) # waits infinitely for a keypress
            if k == ord("q"):
                self.stop()
    
    def push(self):
        """\
:return: the frame after being processed by self.processFrame
Prepare for self.processFrame"""
        
        frame = self.read()
        if frame is None:
            self.stop()
        else:
            return self.processFrame(frame)
    
    def start(self):
        """\
:return: None

Start capturing frames, calling self.processFrame, and display frames"""
        while not self.stopped:
            t0 = clock() # start time
            frame = self.push()
            t1 = clock() # end time
            
            self.pop(frame, t1 - t0)
    
    def stop(self):
        """
:return: None

destroy camera and opened windows when camera stream has ended"""
        self.stopped = True
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # an example of how to use Handler
    class Testing(Handler):
        #self.processFrame(frame) has to return (display_frame)
        def processFrame(self, frame):
            # processing goes here
            for rep in range(10):
                cv2.meanStdDev(frame)
            return frame

    cap = cv2.VideoCapture("concept_test/jiggle_1.mp4")
    h = Testing(cap, isStepped=1)
    h.start()
    

