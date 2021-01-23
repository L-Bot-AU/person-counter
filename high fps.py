import cv2
import imutils
import subprocess
import math
import time

def runCmd(cmd):
    print(subprocess.getoutput(cmd.split(" ")))

runCmd("focusuf\\FocusUF.exe --camera-name IZONE --focus-mode-manual")
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break
    
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"): break
    print(1/(time.time() - prev))
    prev = time.time()

cv2.destroyAllWindows()


"""
normal: 0.028
"""
