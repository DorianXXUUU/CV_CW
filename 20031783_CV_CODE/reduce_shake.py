import numpy as np
import cv2
import sys

def moving_average_filter(curve, radius): 
    # f[k] = (+ ... + c[k-2]+c[k-1]+c[k]+c[k+1]+c[k+2] + ...+ ) / n
    N = 2 * radius + 1
    # Define the filter 
    maf = np.ones(N) / N 
    
    # Add padding to the boundaries 
    paddding_curve = np.lib.pad(curve, (radius, radius), 'edge')
    
    # convolution 
    smoothed_curve = np.convolve(paddding_curve, maf, mode='same') 
    
    # Remove padding 
    smoothed_curve = smoothed_curve[radius: -radius]
    return smoothed_curve

def smooth_trajectory(trajectory): 
    smoothed_trajectory = np.copy(trajectory) 
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:,i] = moving_average_filter(trajectory[:,i], radius = SMOOTHING_RADIUS)
    
    return smoothed_trajectory

# read video
video = cv2.VideoCapture('Test1.MOV')
SMOOTHING_RADIUS = 5

# count the frame number
frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
# frame_num = 450 for test3

# Get width and height of video stream
width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = video.get(cv2.CAP_PROP_FPS)

# output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.avi', fourcc, fps, (int(width), int(height)))

# transformation store array
transformation_matrix = np.zeros((frame_num - 1, 3), np.float32) 

# Read first frame
ret, first_frame = video.read() 
if ret is False:
    sys.exit(1)

# Convert frame to grayscale
previous_grayscale = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# calculate the transform for each frame except the first.
for i in range(frame_num - 2):
    previous_feature_points_pts = cv2.goodFeaturesToTrack(previous_grayscale,
                                     maxCorners=200,
                                     qualityLevel=0.1,
                                     minDistance=30,
                                     blockSize=7)
    # Read next frame
    if_next, current_frame = video.read() 
    if not if_next: 
        break 
    
    # Convert the current read frame to grayscale
    cuurent_grayscale = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY) 
    # calc the optical flow
    current_feature_points_pts, status, err = cv2.calcOpticalFlowPyrLK(previous_grayscale, 
                                                                       cuurent_grayscale, 
                                                                       previous_feature_points_pts, 
                                                                       None,winSize=(15, 15), 
                                                                       maxLevel=2,
                                                                       criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                                                                      )
    
    # Selects good feature points for previous position
    previous_feature_points_pts = previous_feature_points_pts[status == 1]
    current_feature_points_pts = current_feature_points_pts[status == 1]
    
    # calculate the transformation matrix
    transformation_rigid_matrix, rigid_mask = cv2.estimateAffinePartial2D(previous_feature_points_pts, current_feature_points_pts)
    print(motion)
    # transform along x-axis, y-axis and ritation
    dx = transformation_rigid_matrix[0,2]
    dy = transformation_rigid_matrix[1,2]
    da = np.arctan2(transformation_rigid_matrix[1, 0], transformation_rigid_matrix[0, 0])
    
    # save it to transform matrix for each frame.
    transformation_matrix[i] = [dx, dy, da]
    
    # update the previous frame.
    previous_grayscale = cuurent_grayscale
    
# calculate the trajectories
original_trajectory = np.cumsum(transformation_matrix, axis=0)

# smooth the trajectories
smoothed_trajectories = smooth_trajectory(original_trajectory)

# Calculate difference between two tragectory
diff = smoothed_trajectories - trajectory

# Calculate the new transformation
new_transforms = transformation_matrix + diff

# implement the transforms to the original video

# back to beginning
video.set(cv2.CAP_PROP_POS_FRAMES, 0) 
 
# Write n_frames - 1 transformed frames
for i in range(frame_num - 2):
    # Read next frame
    ret, current_frame = video.read() 
    if not ret:
        break
 
    # derive transformations
    dx = new_transforms[i, 0]
    dy = new_transforms[i, 1]
    da = new_transforms[i, 2]

    # Reconstruct transformation matrix accordingly to new values
    # the transformation matrix = cos -sin x
    #                             sin  cos y
    new_transformation_matrix = np.zeros((2,3), np.float32)
    new_transformation_matrix[0,2] = dx
    new_transformation_matrix[1,2] = dy
    new_transformation_matrix[0,0] = np.cos(da)
    new_transformation_matrix[0,1] = -np.sin(da)
    new_transformation_matrix[1,0] = np.sin(da)
    new_transformation_matrix[1,1] = np.cos(da)
 
    # affine wrapping - stablized
    processed_frame = cv2.warpAffine(current_frame, matrix, (int(width), int(height)))

    frame_out = cv2.hconcat([current_frame, processed_frame])

    cv2.imshow("before processing and after processing", frame_out)
    
    # output the frame
    output.write(frame_out)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break


video.release()
output.release()
cv2.destroyAllWindows()