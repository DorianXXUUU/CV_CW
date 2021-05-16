import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import math
import sys

# Load the video
video = cv2.VideoCapture("./Test1.MOV")

w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) 
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Variable for color to draw optical flow track
optical_flow_line_color = (0, 255, 0)

# Take first frame and find corners in it
# if frame is read correctly ret is True
ret, first_frame = video.read()
if ret is False:
    sys.exit(1)

# change the resolution for the video.
first_frame = cv2.resize(first_frame, 
                         (600, 400), 
                         interpolation = cv2.INTER_AREA)

# Converts frame to grayscale
previous_grayscale = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Feature selection
# Finds the corners using Shi-Tomasi method
detected_feature_points = cv2.goodFeaturesToTrack(previous_grayscale, 
                             mask = None, 
                             maxCorners=100,
                             qualityLevel=0.2,
                             minDistance=10,
                             blockSize=7,
                             useHarrisDetector=False
                            )

# record these good feature points
initial_good_feature = detected_feature_points.copy()
# Creates an  drawing
# mask = np.zeros((first_frame.shape[0], first_frame.shape[1], first_frame.shape[2]))
mask = np.zeros_like(first_frame)

while(video.isOpened()):
    # ret = a boolean return value from getting the frame
    # frame = the current frame being projected in the video
    ret, current_frame = video.read()
    if ret is False:
        break
    current_frame = cv2.resize(current_frame, (600, 400), interpolation = cv2.INTER_AREA)
    
    current_grayscale = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    # Calculates sparse optical flow by Lucas-Kanade method
    next_Pts, status, error = cv2.calcOpticalFlowPyrLK(previous_grayscale, current_grayscale, detected_feature_points, 
                                                 None, 
                                                 winSize=(15, 15), 
                                                 maxLevel=2,
                                                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                                                )
    
    # Selects good feature points for previous position
    previous_good_features = detected_feature_points[status == 1]
    # Selects good feature points for next position
    current_good_features = next_Pts[status == 1]
    print(zip(current_good_features, previous_good_features))
    
    # delete the statical feature points
    index = 0
    # control the balance between filter and track
    distance_criteria = 1
    for i, (new_positions, old_positions) in enumerate(zip(current_good_features, previous_good_features)):

        x0, y0 = new_positions.ravel()
        x1, y1 = old_positions.ravel()

        distance = abs(x0 - x1) + abs(y0 - y1)

        if distance > distance_criteria:

            current_good_features[index] = current_good_features[i]
            previous_good_features[index] = previous_good_features[i]
            initial_good_feature[index] = initial_good_feature[i]

            index = index +1
            
    # select dynamic feature points and delete other featrue points
    initial_good_feature = initial_good_feature[:index]
    current_good_features = current_good_features[:index]
    previous_good_features = previous_good_features[:index]
    
    # Draws the optical flow tracks
    for i, (new_positions, old_positions) in enumerate(zip(current_good_features, previous_good_features)):
        # Returns a contiguous flattened array as (x, y) coordinates for new point
        x0, y0 = new_positions.ravel()
        # Returns a contiguous flattened array as (x, y) coordinates for old point
        x1, y1 = old_positions.ravel()
        # Draws line between new and old position with green color and 2 thickness
        mask = cv2.line(mask, (x0, y0), (x1, y1), optical_flow_line_color, 2)
        # Draws filled circle (thickness of -1) at new position with green color and radius of 3
        current_frame = cv2.circle(current_frame, (x0, y0), 2, optical_flow_line_color, -1)
        
    # Overlays the optical flow tracks on the original frame
    output = cv2.add(current_frame, mask)
    cv2.imshow('Sparse optical flow',output)
    
    # Frames are read by intervals of 10 milliseconds. 
    # The programs breaks out of the while loop when the user presses the 'q' key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
    # Updates previous frame
    previous_grayscale = current_grayscale.copy()
    # Updates previous good feature points
    detected_feature_points = current_good_features.reshape(-1, 1, 2)

    if initial_good_feature.shape[0] < 50:
        detected_feature_points = cv2.goodFeaturesToTrack(previous_grayscale, 
                             mask = None, 
                             maxCorners = 100,
                             qualityLevel = 0.2,
                             minDistance = 10,
                             blockSize = 7,
                             useHarrisDetector = False
                            )
        
        initial_good_feature = detected_feature_points.copy()
        

# The following frees up resources and closes all windows
video.release()
cv2.destroyAllWindows()