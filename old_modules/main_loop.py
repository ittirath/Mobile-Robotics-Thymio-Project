import cv2

from CV_aruco_module import get_pose_from_frame, draw_global_path
from global_nav_module import get_global_path

def camera_to_world(frame_size,xy,theta):
  return [xy[0], frame_size[1] - xy[1]], -theta

def path_world_to_camera(frame_size, path_world):
    H = frame_size[1]
    path_cam = path_world.copy()
    path_cam[1, :] = H - path_cam[1, :]
    return path_cam

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

state = "GLOBAL_NAVIGATION"
thymio_start = thymio_theta = goal = polygons = homography = None
global_path = None
frame_size = [77.0e-2,71.0e-2]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received")
        break
    
    if state == "GLOBAL_NAVIGATION":
        thymio_start, thymio_theta, goal, polygons_real_world, homography = get_pose_from_frame(frame, only_thymio=False)
        #global_path_cam = get_global_path(thymio_start, goal, polygons_real_world, plot=False)

        if thymio_start is not None or goal is not None:
            if polygons_real_world is not None or len(polygons_real_world) != 0:  
                thymio_start, thymio_theta = camera_to_world(frame_size, thymio_start, thymio_theta)
                goal, _ = camera_to_world(frame_size, goal, 0)
                for i in range(len(polygons_real_world)):
                    for j in range(len(polygons_real_world[i])):
                        polygon_edge, _ = camera_to_world(frame_size, polygons_real_world[i][j], 0)
                        polygons_real_world[i][j][0] = polygon_edge[0]
                        polygons_real_world[i][j][1] = polygon_edge[1]

        global_path = get_global_path(thymio_start, goal, polygons_real_world, plot=False) # set plot to True to save figure


        if global_path is not None:
            global_path_cam = path_world_to_camera(frame_size, global_path)
            draw_global_path(frame, homography, global_path_cam)

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
