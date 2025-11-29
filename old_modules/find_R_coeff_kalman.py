import cv2
import numpy as np

from CV_aruco_module import get_pose_from_frame

# place thymio aruco with top left corner touching the top left corner of the map i.e. bottom right corner of 0 aruco marker
true_pos = np.array([4.5, 4.5])
# make sure the star is at the top right corner of thymio aruco marker
true_theta = 0.0

thymio_start = thymio_theta = None
x_var = y_var = theta_var = 0 

N = 0 

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received")
        break
    
    thymio_start, thymio_theta = get_pose_from_frame(frame, only_thymio=True)
    if thymio_start is None or thymio_theta is None:
        continue
    x_var += (thymio_start[0] - true_pos[0])**2
    y_var += (thymio_start[1] - true_pos[1])**2
    theta_var += (thymio_theta - true_theta)**2
    N += 1

    cv2.imshow("Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print (" x_var = ",x_var / N)
print (" y_var = ",y_var / N)
print (" theta_var = ",theta_var / N)

cap.release()
cv2.destroyAllWindows()

# TRIAL 1
#  x_var =  0.7844658447707921
#  y_var =  0.9295127986626432
#  theta_var =  0.0004475837995998382  

# TRIAL 2
# x_var =  0.5105208184986603
#  y_var =  0.866067145591887

# # TRIAL 3
#  x_var =  0.5999971125456232
#  y_var =  0.23774834107316864

# average x = 0.6316612586050252
# average y = 0.677776095109233