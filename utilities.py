"""
Uses python threading capabilities to increase frame rate
TODO:
comments
"""

import cv2
from multiprocessing.pool import ThreadPool
from collections import deque
import imutils
import subprocess

def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()

def draw_str(dst, s, target):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

class DummyTask:
    def __init__(self, data):
        self.data = data
    def ready(self):
        return True
    def get(self):
        return self.data

# disable autofocus
def runCmd(cmd):
    print(subprocess.getoutput(cmd.split(" ")))
    
class Handler:
    """wrapper for displaying and processing video"""
    
    def __init__(self, cap, CAMERA_NAME, isThreaded=False, isStepped=False):
        """given cap as a cv2.VideoCapture, handle the high level processing"""
        self.cap = cap
        self.isThreaded = isThreaded
        self.isStepped = isStepped
        self.stopped = False
        
        if not CAMERA_NAME:
            runCmd("focusuf\\FocusUF.exe --list-cameras")
            raise Exception("select a camera")
        runCmd(f"focusuf\\FocusUF.exe --camera-name {CAMERA_NAME} --focus-mode-manual")
        
        self.threadn = cv2.getNumberOfCPUs()
        self.pool = ThreadPool(processes=self.threadn)
        self.pending = deque()
        self.prevFrame = clock()
        
    def pop(self):
        frame, t0 = self.pending.popleft()
        frame = imutils.resize(frame.get(), width=900)
        
        now = clock()
        fps = (now - self.prevFrame) ** -1
        self.prevFrame = now
        
        draw_str(frame, f"latency: {round((now - t0)*1000, 2)} ms", (20, 20))
        draw_str(frame, f"{round(fps, 2)} frames/s", (20, 40))
        cv2.imshow("video", frame)
        
        
        # if the `q` key is pressed, break from the loop
        if not self.isStepped and cv2.waitKey(1) & 0xFF == ord("q"):
            self.stop()
        if self.isStepped:
            cv2.waitKey(0)

    def push(self):
        _ret, frame = self.cap.read()
        if not _ret or frame is None:
            self.stop()
        elif self.isThreaded:
            self.pending.append((
                self.pool.apply_async(self.processFrame, (frame,)),
                clock()
            ))
        else:
            self.pending.append((
                DummyTask(self.processFrame(frame,)),
                clock()
            ))
        
    def start(self):
        """start the stream"""
        try:
            self.processFrame
        except AttributeError:
            raise AttributeError("create self.processFrame(frame, t) which returns (display_frame, t)")
        
        while not self.stopped:
            if self.pending and self.pending[0][0].ready():
                self.pop()
                
            if len(self.pending) < self.threadn:
                self.push()
        
    def stop(self):
        """[called automatically] destroy camera and opened frames when stream has ended"""
        self.stopped = True
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # an example of how to use this
    class Testing(Handler):
        #self.processFrame(frame, t) has to return (display_frame, t)
        def processFrame(self, frame):
            # processing goes here
            for rep in range(10):
                cv2.meanStdDev(frame)
            return frame
    
    h = Testing(cv2.VideoCapture("concept_test/jiggle_1.mp4"))
    h.start()
