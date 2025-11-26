import cv2

from CV_aruco_module import get_pose_from_frame, draw_global_path
from global_nav_module import get_global_path


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

state = "GLOBAL_NAVIGATION"
thymio_start = thymio_theta = goal = polygons = homography = None
global_path = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received")
        break
    
    if state == "GLOBAL_NAVIGATION":
        thymio_start, thymio_theta, goal, polygons, homography = get_pose_from_frame(frame, only_thymio=False)

        global_path = get_global_path(thymio_start, goal, polygons, plot=False) # set plot to True to save figure 
        if global_path is not None:
            draw_global_path(frame, homography, global_path)

        #state = "ROTATE"

    elif state == "ROTATE":
        # kalman -> thymio_start, thymio_theta = get_pose_from_frame(frame, only_thymio = True)
        # verify local nav -> state = "LOCAL_NAVIGATION" 
        # rotate in place 
        # if aligned switch to FORWARD
        pass  
    elif state == "FORWARD":
        # kalman -> thymio_start, thymio_theta = get_pose_from_frame(frame, only_thymio = True)
        # verify local nav -> state = "LOCAL_NAVIGATION" 
        # move forward
        # if reached next waypoint switch to ROTATE
        pass  
    elif state == "LOCAL_NAVIGATION":
        # kalman (technically optional but better to do in order to have continous pose estimation and avoid jumps) 
        # local nav 
        # if no obstacle switch to GLOBAL_NAVIGATION
        pass  

    cv2.imshow("Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # cv2.waitKey(100) # wait 100 ms (NOT SURE -> TO BE TESTED)

cap.release()
cv2.destroyAllWindows()
