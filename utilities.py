import cv2
import imutils
import math
import subprocess

PROCESSING_RES = 720
DISPLAY_RES = 480

def distance(p0, p1):
    """euclidian distanced between 2 coordinates"""
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

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
    
class Handler:
    """wrapper for displaying and processing video"""
    
    def __init__(self, cap, CAMERA_NAME=None, isStepped=False):
        """given cap as a cv2.VideoCapture, handle the high level processing"""
        self.cap = cap
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROCESSING_RES)
        self.isStepped = isStepped
        self.stopped = False
        
        # disable autofocus
        if not CAMERA_NAME and type(cap) != cv2.VideoCapture:
            runCmd("focusuf\\FocusUF.exe --list-cameras")
            raise Exception("select a camera")
        runCmd(f"focusuf\\FocusUF.exe --camera-name {CAMERA_NAME} --focus-mode-manual --exposure-mode-manual")
        
    def read(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        return imutils.resize(frame, height=PROCESSING_RES) # 720p resolution
    
    def imshow(self, windowName, frame):
        cv2.imshow(windowName, imutils.resize(frame, height=DISPLAY_RES))
        
    def pop(self, frame, timeElapsed):
        draw_str(frame, f"latency: {round(timeElapsed * 1000, 3)} ms", (20, 20))
        draw_str(frame, f"{round(1 / timeElapsed, 3)} frames/s", (20, 40))
        self.imshow("video", frame)
        
        if self.isStepped:
            k = cv2.waitKey(0) # actually waits
        else:
            k = cv2.waitKey(1) # doesnt wait
            
        if k == ord("q"):
            self.stop()
        if not self.isStepped and k == ord("p"):
            cv2.waitKey(-1)

    def push(self):
        frame = self.read()
        if frame is None:
            self.stop()
        else:
            return self.processFrame(frame)
        
    def start(self):
        """start the stream"""
        try:
            self.processFrame
        except AttributeError:
            raise AttributeError("create self.processFrame(frame, t) which returns (display_frame, t)")
        
        while not self.stopped:
            t0 = clock()
            frame = self.push()
            t1 = clock()
            
            self.pop(frame, t1 - t0)
        
    def stop(self):
        """[called automatically] destroy camera and opened frames when stream has ended"""
        self.stopped = True
        self.cap.release()
        cv2.destroyAllWindows()

def fpsTest(cap):
    import time
    def img_hash(frame):
        roi = frame[y1 : y2, x1 : x2]
        return hash(roi.tostring())
    
    print("init camera")
    print(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720))
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G')))
    print(cap.set(cv2.CAP_PROP_FPS, 60))
    print(cap.get(cv2.CAP_PROP_FPS))
    
    while not cap.isOpened(): continue
    _ret1, prevTime = cap.read()
    
    img_height, img_width = prevTime.shape[:2]
    print(img_height, img_width)
    img_center_x, img_center_y = int(img_width / 2), int(img_height / 2)

    roi_size = 8
    x1, y1 = (img_center_x - roi_size, img_center_y - roi_size)
    x2, y2 = (img_center_x + roi_size, img_center_y + roi_size)

    start = time.time()
    NUM_TRIALS = 100
    for i in range(NUM_TRIALS):
        while True:
            _ret2, currFrame = cap.read()
            if img_hash(prevTime) != img_hash(currFrame):
                break
        prevTime = currFrame
    
    print(NUM_TRIALS / (time.time() - start))
    
if __name__ == '__main__':
    # an example of how to use this
    """class Testing(Handler):
        #self.processFrame(frame, t) has to return (display_frame, t)
        def processFrame(self, frame):
            # processing goes here
            for rep in range(10):
                cv2.meanStdDev(frame)
            return frame

    cap = cv2.VideoCapture("concept_test/jiggle_1.mp4")
    h = Testing(cap, isStepped=1)
    h.start()"""
    
    cap = cv2.VideoCapture(2) # this takes a long time for c922
    fpsTest(cap)

