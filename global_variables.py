import numpy as np
import cv2

# COMPUTER VISION VARIABLES ----------------------------------------

# Real-world coordinates of the 8 corners of the aruco markers
# Order must match the corner order from ArUco: TL, TR, BR, BL.
# Top-left marker
pts_world_TL = np.array([
    [-6.5, -6.5],   # top-left
    [ 0.0, -6.5],   # top-right
    [ 0.0,  0.0],   # bottom-right
    [-6.5,  0.0],   # bottom-left
], dtype=np.float32)

# Top-right marker
pts_world_TR = np.array([
    [77.0, -6.5],   # top-left
    [83.5, -6.5],   # top-right
    [83.5,  0.0],   # bottom-right
    [77.0,  0.0],   # bottom-left
], dtype=np.float32)

# Bottom-right marker
pts_world_BR = np.array([
    [77.0, 71.0],   # top-left
    [83.5, 71.0],   # top-right
    [83.5, 77.5],   # bottom-right
    [77.0, 77.5],   # bottom-left
], dtype=np.float32)

# Bottom-left marker
pts_world_BL = np.array([
    [-6.5, 71.0],   # top-left
    [0.0, 71.0],   # top-right
    [0.0, 77.5],   # bottom-right
    [-6.5, 77.5],   # bottom-left
], dtype=np.float32)

pts_world = np.vstack([pts_world_TL,pts_world_TR, pts_world_BR,pts_world_BL])     # shape (8,2)

# ArUco dictionary (DICT_4X4_50)
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# aruco_params = cv2.aruco.DetectorParameters()
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector_params = cv2.aruco.DetectorParameters()

# KALMAN FILTER VARIABLES ----------------------------------------

dt = 0.3 # time between EKF calls [s]
alpha = 0 # linear gain in uncertainty
threshold = 11 # cutoff of the Mahalanobis distance for measurement outlier rejection

var_v_l = 1 # variance in the left wheel linear speed measurements [cm2/s2]
var_v_r = 1 # variance in the right wheel linear speed measurements [cm2/s2]

var_x_cam = 0.005 # variance in the x coordinate camera measurement [cm2]
var_y_cam = 0.005 # variance in the y coordinate camera measurement [cm2]
var_theta_cam = 0.005 # variance in the heading camera measurement [rad2]


# GLOBAL NAVIGATION VARIABLES ----------------------------------------

scale_factor = 4.0  # 2x bigger obstacles

# LOCAL NAVIGATION VARIABLES ----------------------------------------

BASE_SPEED = 100
THRESH_ENTRY = 1000
FIVE_CM = 5 # number of iterations to advance 5 cm for BASE_SPEED 100 dt = 0.1
NINETY_DEGREES = 12 # number of iterations to turn 90 degrees for BASE_SPEED 100 dt = 0.1
#MARGIN = 1  # uncomment if margin is used

# MOTION CONTROL VARIABLES ----------------------------------------
N = 100
forward_speed = 2e-2
eps_theta = np.pi/50 # tolerance for the robot's heading to be considered as aligned with the waypoint
#eps_d = 6.5e-2 # tolerance for the robot's position to be considetred as on the waypoint
eps_d = 4.5

ROTATION_SPEED = 50 # speed for rotation in place (deg/s)
FORWARD_SPEED = 80  # speed for forward motion (cm/s)
MAX_ROT_CMD = 70          # adjust to your Thymio (max motor command)
K_ROT       = MAX_ROT_CMD / np.pi  # full speed at 180° error

# OTHER GLOBAL VARIABLES ----------------------------------------
frame_size = [77.0e-2,71.0e-2]
wheel_spacing = 10 # wheel spacing [m]
KIDNAPPING_THRESHOLD = 15  # threshold distance [m] to detect kidnapping