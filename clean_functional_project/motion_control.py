import numpy as np

from global_variables import *

# MOTION CONTROL FUNCTIONS ----------------------------------------

def forward_P_regulator(X, waypoint, v_forward):
    """
    Proportional regulator for forward motion towards a waypoint.
    X         : [x, y, theta]
    waypoint  : [x_wp, y_wp]
    v_forward : nominal forward command
    """
    theta_des = np.arctan2(waypoint[1] - X[1], waypoint[0] - X[0])
    delta_theta = (theta_des - X[2] + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

    rot_cmd = K_ROT * delta_theta
    rot_cmd = np.clip(rot_cmd, -MAX_ROT_CMD, MAX_ROT_CMD)

    # generalization of your ROTATE pattern with a forward bias
    v_r = v_forward - rot_cmd
    v_l = v_forward + rot_cmd
    return int(v_r), int(v_l)