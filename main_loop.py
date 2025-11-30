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
waypoints = None
waypoint_index = 1 # 1 because 0 is the initial position
X_array = [] # vector state logging over time
X_camera_array = [] # vector state from camera logging over time
X = None
P = np.array([[var_x_cam,0,0],[0,var_y_cam,0],[0,0,var_theta_cam]])

prox_values = None
timer_start = 0
previous_wall_side = None 

# def enter_local_navigation():
#     reset_local_state()
#     state = "LOCAL_NAVIGATION"

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

        waypoints = get_global_path(thymio_start, goal, polygons_real_world, plot=False)
        if waypoints is None:
            # no path found -> stop and wait
            print("no path")
            apply_motor_commands(0, 0)
            cv2.imshow("Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # initialize state X for kalman filter
        X = np.array([thymio_start[0], thymio_start[1], thymio_theta])
        waypoint_index = 1
        state = "ROTATE"

    else:
        # ROTATE / FORWARD: only need Thymio pose (kalman should be here normally)
        thymio_start, thymio_theta = get_pose_from_frame(frame, only_thymio=True)
        if thymio_start is not None and thymio_theta is not None:
            # UPDATE KALMAN FILTER HERE --------------------------------------------------------------------------
            v_r_motor, v_l_motor = get_motor_speeds()
            X, P = update_EKF(thymio_start, thymio_theta, v_r_motor, v_l_motor, X, P, frame)
            #print(f"Kalman pose: x={X[0]:.3f} m, y={X[1]:.3f} m, theta={X[2]:.3f} rad")  # debug

            X_array.append(X.copy()) # copy to avoid corruption
            # Debug: override kalman with camera measurement
            # X[0] = thymio_start[0]
            # X[1] = thymio_start[1]
            # X[2] = thymio_theta
            X_camera_array.append(np.array([thymio_start[0], thymio_start[1], thymio_theta]))
            
            
            # CHECK FOR KIDNAPPING HERE --------------------------------------------------------------------------
            if len(X_array) >= 2:
                x_diff = X_array[-1][0] - X_array[-2][0]
                y_diff = X_array[-1][1] - X_array[-2][1]
            else:
                x_diff = 0
                y_diff = 0
            dist_moved = np.linalg.norm([x_diff, y_diff])
            if dist_moved > KIDNAPPING_THRESHOLD:  # threshold for kidnapping detection
                print("Kidnapping detected")
                state = "GLOBAL_NAVIGATION"

            # CHECK FOR LOCAL NAVIGATION TRIGGER HERE --------------------------------------------------------------------------
             # Check Obstacle
            prox_values = get_prox_sensors()
            if check_obstacle_trigger(prox_values):
                local_nav_log("Obstacle detected! Switching to Local.")
                # enter_local_navigation()
                reset_local_state()
                state = "LOCAL_NAVIGATION" 

    if state == "ROTATE":
    
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

        x_diff = waypoints[0,waypoint_index] - X[0]
        y_diff = waypoints[1,waypoint_index] - X[1]
        dist_to_waypoint = np.linalg.norm([x_diff, y_diff])

        #print(f"Distance to waypoint: {dist_to_waypoint:.3f} m") # debug

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
        print("State: LOCAL NAV")
            
        # Logic
        left_speed, right_speed = local_nav_update(prox_values)
        
        # Act
        apply_motor_commands(left_speed, right_speed)

        # Exit Condition
        if current_state == "GLOBAL" and max(prox_values[:5]) < THRESH_ENTRY:
            local_nav_log("Obstacle cleared. Returning to Global.")
            stop_robot()
            state = "GLOBAL_NAVIGATION" 

    if waypoints is not None and homography is not None:
        draw_global_path(frame, homography, waypoints)
        
    cv2.imshow("Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elapsed = time.time() - current_time
    if elapsed < dt: 
        time.sleep(dt - elapsed)

print("Thymio reached goal")

cap.release()
cv2.destroyAllWindows()

# PLOTTING THE KALMAN FILTER RESULTS ----------------------------------------

X_np      = np.array(X_array)         # shape (N, 3)
Xcam_np   = np.array(X_camera_array)  # shape (N, 3)
t         = np.arange(len(X_np)) * dt # or np.arange(len(X_np))

plt.figure(figsize=(10, 8))

# x
plt.subplot(3, 1, 1)
plt.plot(t, X_np[:, 0],    label="EKF x")
plt.plot(t, Xcam_np[:, 0], label="Camera x", linestyle="--")
plt.ylabel("x")
plt.legend()
plt.grid(True)

# y
plt.subplot(3, 1, 2)
plt.plot(t, X_np[:, 1],    label="EKF y")
plt.plot(t, Xcam_np[:, 1], label="Camera y", linestyle="--")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# theta
plt.subplot(3, 1, 3)
plt.plot(t, X_np[:, 2],    label="EKF theta")
plt.plot(t, Xcam_np[:, 2], label="Camera theta", linestyle="--")
plt.xlabel("time [s]")
plt.ylabel("theta [rad]")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()