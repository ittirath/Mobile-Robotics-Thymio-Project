import warnings
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt

from global_variables import *
from computer_vision import *
from kalman import *
from global_navigation import *
from local_navigation import *
from motion_control import *

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

# Global variables
state = "GLOBAL_NAVIGATION"
thymio_start = thymio_theta = goal = polygons = homography = None
global_path = None
is_aligned = False
is_reached = False
is_locally_navigating = False
waypoint_index = 1 # 1 because 0 is the initial position

# Variables for plotting
X_array = [] # vector state logging over time
nav_type_array = [] # navigation type logging over time (-1: local navigation, 0: (global) pure rotation/alignement with waypoint, 1: (global) forwards motion + heading PD)

ROTATION_SPEED = 10 # speed for rotation in place (deg/s)
FORWARD_SPEED = 10  # speed for forward motion (cm/s)
MAX_ROT_CMD = 150          # adjust to your Thymio (max motor command)
K_ROT       = MAX_ROT_CMD / np.pi  # full speed at 180° error

while True:
    current_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Frame not received")
        break
    
    if state == "GLOBAL_NAVIGATION":
        thymio_start, thymio_theta, goal, polygons_real_world, homography = get_pose_from_frame(frame, only_thymio=False)
        
        if thymio_start is not None and goal is not None and polygons_real_world is not None:
            thymio_start, thymio_theta = camera_to_world(frame_size, thymio_start, thymio_theta)
            goal, _ = camera_to_world(frame_size, goal, 0)
            for i in range(len(polygons_real_world)):
                for j in range(len(polygons_real_world[i])):
                    polygon_edge, _ = camera_to_world(frame_size, polygons_real_world[i][j], 0)
                    polygons_real_world[i][j][0] = polygon_edge[0]
                    polygons_real_world[i][j][1] = polygon_edge[1]

        waypoints = get_global_path(thymio_start, goal, polygons_real_world, plot=False) # set plot to True to save figure 
        
        # waypoints = get_objective_waypoints(thymio_start, goal, polygons_real_world)
        # target_waypoint = waypoints.pop()

        # Initial pose measurement (with camera)
        if thymio_start is not None or thymio_theta is not None:
            X = np.array([thymio_start[0], thymio_start[1], thymio_theta])
            P = np.array([[var_x_cam,0,0],[0,var_y_cam,0],[0,0,var_theta_cam]])
            state = "ROTATE"


    elif state == "ROTATE":
        v_r_motor, v_l_motor = get_motor_speeds()
        X, P = update_EKF(v_r_motor, v_l_motor, X, P, frame)
        X_array.append(X)
        
        # theta_des = np.arctan2(global_path[1,waypoint_index]-X[1], global_path[0,waypoint_index]-X[0])
        # delta_theta = (theta_des - X[2] + np.pi) % (2*np.pi) - np.pi
        #delta_theta = wrap_to_pi(theta_des - X[2])
        
        X[2] = wrap_to_pi(X[2]) # wrap current theta to ]-pi,pi]
        theta_des = np.arctan2(waypoints[1, waypoint_index]-X[1], waypoints[0, waypoint_index]-X[0])
        delta_theta = (theta_des - X[2] + np.pi) % (2*np.pi) - np.pi

        print(f"theta_des={theta_des:.3f}, theta={X[2]:.3f}, delta={delta_theta:.3f}") # debug 

        if abs(delta_theta) < eps_theta:
            apply_motor_commands(0, 0)
            state = "FORWARD"
        else:
            apply_motor_commands(ROTATION_SPEED,-ROTATION_SPEED)
        
    elif state == "FORWARD":
        v_r_motor, v_l_motor = get_motor_speeds()
        X, P = update_EKF(v_r_motor, v_l_motor, X, P, frame)
        X_array.append(X)

        x_diff = waypoints[0,waypoint_index] - X[0]
        y_diff = waypoints[1,waypoint_index] - X[1]
        dist_to_waypoint = np.linalg.norm([x_diff, y_diff])
        
        if dist_to_waypoint < eps_d:
            apply_motor_commands(0,0)
            waypoint_index += 1
            if waypoint_index >= waypoints.shape[1]:
                break
            state = "ROTATE" # reached waypoint
        else:
            apply_motor_commands(FORWARD_SPEED, FORWARD_SPEED) # move forward

    elif state == "LOCAL_NAVIGATION":
        v_r_motor, v_l_motor = get_motor_speeds()
        X, P = update_EKF(v_r_motor, v_l_motor, X, P, frame)
        X_array.append(X)
        
        # local nav 
        # if no obstacle switch to GLOBAL_NAVIGATION
        pass  

    if waypoints is not None:
        waypoints_cam = path_world_to_camera(frame_size, waypoints)
        draw_global_path(frame, homography, waypoints_cam)
        
    cv2.imshow("Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    # cv2.waitKey(100) # wait 100 ms (NOT SURE -> TO BE TESTED)
    while True:
        if time.time() - current_time >= dt: # at most 10 Hz (not too fast)
            break

cap.release()
cv2.destroyAllWindows()
