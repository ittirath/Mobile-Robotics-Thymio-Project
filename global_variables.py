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

dt = 0.1 # time between EKF calls [s]
alpha = 0 # linear gain in uncertainty
threshold = 11 # cutoff of the Mahalanobis distance for measurement outlier rejection

var_v_l = 1e-2 # variance in the left wheel linear speed measurements [m2/s2]
var_v_r = 1e-2 # variance in the right wheel linear speed measurements [m2/s2]

var_x_cam = 1e-4 # variance in the x coordinate camera measurement [m2]
var_y_cam = 1e-4 # variance in the y coordinate camera measurement [m2]
var_theta_cam = 1e-2 # variance in the heading camera measurement [rad2]


# GLOBAL NAVIGATION VARIABLES ----------------------------------------

scale_factor = 2.5  # 2x bigger obstacles

# LOCAL NAVIGATION VARIABLES ----------------------------------------

# Constatns
THRESH_ENTRY = 2000      # Threshold to enter local nav
THRESH_WALL_LOST = 400   # Threshold for "losing" the wall
TARGET_DIST = 2000       # Distance to the wall of the obstacle
BASE_SPEED = 100
P_GAIN = 0.1

# Times
CLEARANCE_DURATION = 1.5 # Time to turn the corner (step forward)
TURN_DURATION = 1.5      # Time to turn

# Debug flag to enable debug print
DEBUG_PRINT = True

# State variables at module level (!= main_loop)
current_state = "GLOBAL"
timer_start = 0
previous_wall_side = None 


# MOTION CONTROL VARIABLES ----------------------------------------
N = 100
forward_speed = 2e-2
eps_theta = np.pi/50 # tolerance for the robot's heading to be considered as aligned with the waypoint
#eps_d = 6.5e-2 # tolerance for the robot's position to be considetred as on the waypoint
eps_d = 3.5

ROTATION_SPEED = 50 # speed for rotation in place (deg/s)
FORWARD_SPEED = 80  # speed for forward motion (cm/s)
MAX_ROT_CMD = 60          # adjust to your Thymio (max motor command)
K_ROT       = MAX_ROT_CMD / np.pi  # full speed at 180° error

# OTHER GLOBAL VARIABLES ----------------------------------------
frame_size = [77.0e-2,71.0e-2]
wheel_spacing = 10e-2 # wheel spacing [m]
KIDNAPPING_THRESHOLD = 15  # threshold distance [m] to detect kidnapping