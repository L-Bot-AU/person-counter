import subprocess
import random
import imutils
import math
import cv2

PROCESSING_RES = 720
DISPLAY_RES = 480

def distance(p0, p1):
    """euclidian distanced between 2 coordinates"""
    return math.sqrt((p0[0] - p1[0])*(p0[0] - p1[0]) + (p0[1] - p1[1])*(p0[1] - p1[1]))

def avgPoint(p0, p1):
    return (p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2

def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()

def draw_str(frame, s, target):
    x, y = target
    
    # shaddow
    cv2.putText(frame, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    # text
    cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

def runCmd(cmd):
    print(subprocess.getoutput(cmd.split(" ")))


def genColour(seed):
    random.seed(seed)
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )

class Handler:
    """\
wrapper for displaying and processing video
handles the high level processing"""
    
    def __init__(self, cap, CAMERA_NAME=None, isStepped=False, seekTime=None):
        """\
:param cap: cv2.VideoCapture object
:param CAMERA_NAME: camera name e.g. "c922 Pro Stream Webcam"
:param isStepped: debugger flag indicating whether you progress through frames one at a time on keypress
:param seekTime: jump to a particular time in the video, given in seconds
:return: None

initialises capture, set focus mode to auto
"""
        
        # initialise capture
        self.cap = cap
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROCESSING_RES)
        if seekTime:
            cap.set(cv2.CAP_PROP_POS_MSEC, seekTime*1000)
        self.isStepped = isStepped
        self.stopped = False
        
        # set focus mode to auto
        if not CAMERA_NAME and type(cap) != cv2.VideoCapture:
            runCmd("focusuf\\FocusUF.exe --list-cameras")
            raise Exception("select a camera")
        runCmd(f"focusuf\\FocusUF.exe --camera-name {CAMERA_NAME} --focus-mode-manual --exposure-mode-manual")
    
    def read(self):
        """\
:return: the most recent image captured from self.cap, scaled to PROCESSING_RES"""
        
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        return imutils.resize(frame, height=PROCESSING_RES) # 720p resolution
    
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
        
        draw_str(frame, f"latency: {round(timeElapsed * 1000, 3)} ms", (20, 20))
        draw_str(frame, f"{round(1 / timeElapsed, 3)} frames/s", (20, 40))
        self.imshow("video", frame)
        
        if self.isStepped:
            k = cv2.waitKey(0) # waits infinitely for a keypress
        else:
            k = cv2.waitKey(1) # wait 1ms for a keypress
            
        if k == ord("q"):
            self.stop()
        if not self.isStepped and k == ord("p"):
            cv2.waitKey(0) # waits infinitely for a keypress
            
    
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
        try:
            self.processFrame
        except AttributeError:
            raise NameError("requires a function self.processFrame(frame) which returns displayFrame")
        
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
    

