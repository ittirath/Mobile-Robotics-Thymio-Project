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
theta_des = 0.0
global_path = None
is_aligned = False
is_reached = False
is_locally_navigating = False
waypoint_index = 1 # 1 because 0 is the initial position

# Variables for plotting
X_array = [] # vector state logging over time
nav_type_array = [] # navigation type logging over time (-1: local navigation, 0: (global) pure rotation/alignement with waypoint, 1: (global) forwards motion + heading PD)


while True:
    current_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Frame not received")
        break
    
    if state == "GLOBAL_NAVIGATION":
        thymio_start, thymio_theta, goal, polygons_real_world, homography = get_pose_from_frame(frame, only_thymio=False)
        if thymio_start is None or thymio_theta is None or goal is None:
            cv2.imshow("Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue  # wait for a usable frame
        # if thymio_start is not None and goal is not None and polygons_real_world is not None:
        #     thymio_start, thymio_theta = camera_to_world(frame_size, thymio_start, thymio_theta)
        #     goal, _ = camera_to_world(frame_size, goal, 0)
        #     for i in range(len(polygons_real_world)):
        #         for j in range(len(polygons_real_world[i])):
        #             polygon_edge, _ = camera_to_world(frame_size, polygons_real_world[i][j], 0)
        #             polygons_real_world[i][j][0] = polygon_edge[0]
        #             polygons_real_world[i][j][1] = polygon_edge[1]

        waypoints = get_global_path(thymio_start, goal, polygons_real_world, plot=False)
        if waypoints is None:
            # no path found → stop and wait
            apply_motor_commands(0, 0)
            cv2.imshow("Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # initialize state X from camera
        X = np.array([thymio_start[0], thymio_start[1], thymio_theta])
        waypoint_index = 1
        state = "ROTATE"

    else:
        # ROTATE / FORWARD: only need Thymio pose (kalman should be here normally)
        thymio_start, thymio_theta = get_pose_from_frame(frame, only_thymio=True)
        if thymio_start is not None and thymio_theta is not None:
            X[0] = thymio_start[0]
            X[1] = thymio_start[1]
            X[2] = thymio_theta
            print("Hello")


    if state == "ROTATE":
        # v_r_motor, v_l_motor = get_motor_speeds()
        # X, P = update_EKF(v_r_motor, v_l_motor, X, P, frame)
        # X_array.append(X)
        
        # theta_des = np.arctan2(global_path[1,waypoint_index]-X[1], global_path[0,waypoint_index]-X[0])
        # delta_theta = (theta_des - X[2] + np.pi) % (2*np.pi) - np.pi
        #delta_theta = wrap_to_pi(theta_des - X[2])
        
            theta_des = np.arctan2(waypoints[1, waypoint_index]-X[1], waypoints[0, waypoint_index]-X[0])
            delta_theta = (theta_des - X[2] + np.pi) % (2*np.pi) - np.pi

            #print(f"theta_des={theta_des:.3f}, theta={X[2]:.3f}, delta={delta_theta:.3f}") # debug 
            
            if abs(delta_theta) < eps_theta:
                apply_motor_commands(0, 0)
                state = "FORWARD"
            elif delta_theta > 0:
                apply_motor_commands(-ROTATION_SPEED,ROTATION_SPEED)
            else:
                apply_motor_commands(ROTATION_SPEED, -ROTATION_SPEED)
        
    elif state == "FORWARD":
        # v_r_motor, v_l_motor = get_motor_speeds()
        # X, P = update_EKF(v_r_motor, v_l_motor, X, P, frame)
        # X_array.append(X)

        x_diff = waypoints[0,waypoint_index] - X[0]
        y_diff = waypoints[1,waypoint_index] - X[1]
        dist_to_waypoint = np.linalg.norm([x_diff, y_diff])
        print(f"Distance to waypoint: {dist_to_waypoint:.3f} m") # debug

        if dist_to_waypoint < eps_d:
            apply_motor_commands(0,0)
            waypoint_index += 1
            if waypoint_index >= waypoints.shape[1]:
                break
            state = "ROTATE" # reached waypoint
        else:
            #apply_motor_commands(FORWARD_SPEED, FORWARD_SPEED) # move forward
            v_r, v_l = forward_P_regulator( X, waypoints[:, waypoint_index], FORWARD_SPEED)
            apply_motor_commands(v_r, v_l)

    elif state == "LOCAL_NAVIGATION":
        # v_r_motor, v_l_motor = get_motor_speeds()
        # X, P = update_EKF(v_r_motor, v_l_motor, X, P, frame)
        # X_array.append(X)

        # local nav 
        # if no obstacle switch to GLOBAL_NAVIGATION
        pass  

    if waypoints is not None and homography is not None:
        # waypoints_cam = path_world_to_camera(frame_size, waypoints)
        # draw_global_path(frame, homography, waypoints_cam)
        draw_global_path(frame, homography, waypoints)
        
    cv2.imshow("Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    # cv2.waitKey(100) # wait 100 ms (NOT SURE -> TO BE TESTED)
    # while True:
    #     if time.time() - current_time >= dt: # at most 10 Hz (not too fast)
    #         break
    elapsed = time.time() - current_time
    if elapsed < dt:
        time.sleep(dt - elapsed)

print("Thymio reached goal")
cap.release()
cv2.destroyAllWindows()
