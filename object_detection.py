import cv2
import time
import imutils
import argparse
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream
import os
folder = os.path.dirname(os.path.abspath(__file__))



nn = cv2.dnn.readNetFromCaffe(folder+"/SSD_MobileNet_prototxt.txt", folder+"/SSD_MobileNet.caffemodel")

def detect_object(frame):  
    
    labels = ["background", "aeroplane", "bicycle", "bird", 
    "boat","bottle", "bus", "car", "cat", "chair", "cow", 
    "diningtable","dog", "horse", "motorbike", "person", "pottedplant", 
    "sheep","sofa", "train", "tvmonitor"]
        
    
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]

    #Converting Frame to Blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
    	0.007843, (300, 300), 127.5)

    #Passing Blob through network to detect and predict
    nn.setInput(blob)
    detections = nn.forward()

    return detections, labels


