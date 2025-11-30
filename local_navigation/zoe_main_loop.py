import cv2
import numpy as np
from tdmclient import ClientAsync, aw
# from CV_aruco_module import get_pose_from_frame, draw_global_path
# from global_nav_module import get_global_path
import local_nav_module 


# Robot connection
print("Connecting to Thymio...")
client = ClientAsync()
node = aw(client.wait_for_node())
aw(node.lock())
print("Thymio connected and locked.")

# Local nav
def enter_local_navigation():
    # reset function used to reset module internal state
    global state
    local_nav_module.reset_local_state()
    state = "LOCAL_NAVIGATION"
    set_lights_on()
    local_nav_module.local_nav_log("*** Enetring local nav ***")

# utils i'm using to stop the robot when coming out of local nav
def set_motors(left, right):
    """
    Sends motor commands safely.
    Checks if the library returned a task before waiting, 
    preventing the 'NoneType' crash.
    """
    v = {
        "motor.left.target": [int(left)],
        "motor.right.target": [int(right)]
    }
    # CRITICAL FIX: Capture result and check for None
    result = node.send_set_variables(v)
    if result is not None:
        aw(result)

def set_lights_on():
    v = {"leds.top": [0, 0, 32]} # Blue
    result = node.send_set_variables(v)
    if result is not None:
        aw(result)

def set_lights_off():
    v = {"leds.top": [0, 0, 0]} # Off
    result = node.send_set_variables(v)
    if result is not None:
        aw(result)

def stop_robot():
    set_motors(0, 0)

# Camera
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Could not open webcam")
#     exit()

state = "ROTATE"
thymio_start = thymio_theta = goal = polygons = homography = None
global_path = None

try:
    while True:
        # ret, frame = cap.read()
        # if not ret:
        #     print("Frame not received")
        #     break
        aw(client.sleep(0.1))
        wait_task = node.wait_for_variables({"prox.horizontal"})
        if wait_task is not None:
            aw(wait_task)
        #print("acquired sensor values")
        #acquiring prox sensors
        prox_values = list(node.v.prox.horizontal)
        #print("converted into a list")

        print(f"Max Front Sensor: {max(prox_values[:5])}")

        if state == "GLOBAL_NAVIGATION":
            print("stato GLOBAL NAV")
            # thymio_start, thymio_theta, goal, polygons, homography = get_pose_from_frame(frame, only_thymio=False)
            # global_path = get_global_path(thymio_start, goal, polygons, plot=False) # set plot to True to save figure
            # if global_path is not None:
            #     draw_global_path(frame, homography, global_path)
            #     # state = "ROTATE"
            stop_robot()
        
        elif state == "ROTATE":
            print("stato ROTATE")
            # kalman -> thymio_start, thymio_theta = get_pose_from_frame(frame, only_thymio = True)
            # verify local nav -> state = "LOCAL_NAVIGATION"
            if local_nav_module.check_obstacle_trigger(prox_values):
                local_nav_module.local_nav_log("obstacle detected during ROTATE! Switching to local nav")
                enter_local_navigation()
                continue # to skip the rest of the loop
            # rotate in place 
            # if aligned switch to FORWARD
            set_motors(100,100)

        elif state == "FORWARD":
            print("stato FORWARD")
            # kalman -> thymio_start, thymio_theta = get_pose_from_frame(frame, only_thymio = True)
            # verify local nav -> state = "LOCAL_NAVIGATION"
            if local_nav_module.check_obstacle_trigger(prox_values):
                local_nav_module.local_nav_log("Obstacle detected during FORWARD! Switching to local nav")
                stop_robot() # maybe we can stop before switching? We can also don't
                enter_local_navigation()
                continue 
            # move forward
            # if reached next waypoint switch to ROTATE
            set_motors(100,100)

        elif state == "LOCAL_NAVIGATION":
            print("stato LOCAL NAV")
            # we compute the speeds for local nav
            left_speed, right_speed = local_nav_module.local_nav_update(prox_values)
            set_motors(left_speed,right_speed)
            # Are we done?
            # returning to global only if the module is reset & the path is clear
            if local_nav_module.current_state == "GLOBAL" and max(prox_values[:5]) < local_nav_module.THRESH_ENTRY:
                local_nav_module.local_nav_log("Obstacle cleared, returning to global nav")
                stop_robot()
                state = "GLOBAL_NAVIGATION"
                set_lights_off()

        
        
        # cv2.imshow("Feed", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

except KeyboardInterrupt:
    print("Stopping...")

finally:
    stop_robot()
    aw(node.unlock())
    # cap.release()
    # cv2.destroyAllWindows()
    # we turn the leds off
    set_lights_off
    
    print("Shutdown")