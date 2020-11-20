import math
import cv2
import numpy as np
import logging
import time
import imutils

#global variables
width = 0
height = 0
EntranceCounter = 0
ExitCounter = 0
LineWidth = 5
MinContourArea = 20000  #Adjust ths value according to your usage
MovementTolerance = 100
BinarizationThreshold = 100  #Adjust ths value according to your usage


def ProcessGrayFrame(frame):
    #gray-scale convertion and Gaussian blur filter applying
    GrayFrame = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    GausFrame = cv2.GaussianBlur(GrayFrame, (21, 21), 0)
    return GausFrame

def ProcessThreshFrame(frame):
    GrayFrame = ProcessGrayFrame(frame)
    #Background subtraction and image binarization
    FrameDelta = cv2.absdiff(ReferenceFrame, GrayFrame)
    FrameThresh = cv2.threshold(FrameDelta, BinarizationThreshold, 255, cv2.THRESH_BINARY)[1]
    FrameDilate = cv2.dilate(FrameThresh, None, iterations=3)#Dilate image and find all the contours
    return FrameDilate

# This function compares 2 contours to see if they could possibly be the same contours
def ContourCompare(contour1, contour2):
    (x, y) = contour2[0]
    (cx, cy) = contour1[0]
    distance = math.sqrt((x-cx)**2+(y-cy)**2)
    #print(str(distance))
    return distance < MovementTolerance

# This function compares 1 coutour with a list and returns the first match
def ContourTrack(contour, contourlist):
    if len(contourlist) == 0: # if the list is empty, then return -1
        return -1

    #check all the items for a match
    for x in range(0, len(contourlist)):
        if ContourCompare(contour, contourlist[x]):
            return x

    return -1

#start capturing footage
#camera = cv2.VideoCapture(0)
camera = cv2.VideoCapture("concept_test/normal_2.mp4")

BinaryFrame1 = np.zeros((1080, 1920), dtype=np.uint8)
ContourData = []
ActiveContours = []

#The webcam maybe get some time / captured frames to adapt to ambience lighting. For this reason, some frames are grabbed and discarted.
for i in range(0,20):
    (grabbed, Frame) = camera.read()
    ReferenceFrame = ProcessGrayFrame(Frame)

#encapsulate infinite while block inside a try statement
while True:
    grabbed, Frame = camera.read()
    #if cannot grab a frame, this program ends here.
    ch = cv2.waitKey(1)
    if not grabbed or ch == ord("q"):
        break
    
    height = np.size(Frame,0)
    width = np.size(Frame,1)
    
    BinaryFrame2 = ProcessThreshFrame(Frame)

    BinaryFrameCombo = cv2.bitwise_or(BinaryFrame1, BinaryFrame2)

    # combine the last two frames and fine the contours
    _, cnts, _ = cv2.findContours(BinaryFrameCombo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    BinaryFrame1 = BinaryFrame2

    #plot reference lines (entrance and exit lines)
    cv2.line(Frame, (0,int(height/2)), (int(width),int(height/2)), (255, 0, 0), LineWidth)

    #check all found countours
    for c in cnts:
        #if a contour has small area, it'll be ignored
        if cv2.contourArea(c) < MinContourArea:
            continue

        #draw an rectangle "around" the object
        (x, y, w, h) = cv2.boundingRect(c)
        rectBounds = (x, y, w, h)
        cv2.rectangle(Frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #find object's centroid
        CoordXCentroid = (x+x+w)/2
        CoordYCentroid = (y+y+h)/2
        ObjectCentroid = (int(CoordXCentroid),int(CoordYCentroid))
        cv2.circle(Frame, ObjectCentroid, 1, (0, 0, 0), 5)
        cv2.circle(Frame, ObjectCentroid, MovementTolerance, (255,255,255), 5)

        crossedBoundary = False

        #create new tuple of object
        ContourTuple = (ObjectCentroid, crossedBoundary)

        #compare to list
        cResult = ContourTrack(ContourTuple, ContourData)

        if cResult is -1:
            ActiveContours.append(ContourTuple)
        else: # replace the previous contour data with the new contour data
            #Check to see if you crossed the line in the middle
            if ContourData[cResult][1]: #tracked object has already crossed the line
                ContourTuple = (ContourTuple[0], True)
            elif (ContourData[cResult][0][1] >= height/2 and ContourTuple[0][1] <= height/2):
                if not ContourData[cResult][1]:
                    EntranceCounter = EntranceCounter + 1
                    ContourTuple = (ContourTuple[0], True)
                    #lgging.info("Entrances, " + str(EntranceCounter))
            elif (ContourData[cResult][0][1] <= height/2 and ContourTuple[0][1] >= height/2):
                if not ContourData[cResult][1]:
                    ExitCounter = ExitCounter + 1
                    ContourTuple = (ContourTuple[0], True)
                    #logging.info("Exits, " + str(ExitCounter))

            ActiveContours.append(ContourTuple)


    ContourData = ActiveContours
    ActiveContours = []

    #Write entrance and exit counter values on frame and shows it
    cv2.putText(Frame, "Entrances: {}".format(str(EntranceCounter)), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 1), 2)
    cv2.putText(Frame, "Exits: {}".format(str(ExitCounter)), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Original Frame", imutils.resize(Frame, width=800))
    #time.sleep(1)
    cv2.waitKey(1);

camera.release()
cv2.destroyAllWindows()
