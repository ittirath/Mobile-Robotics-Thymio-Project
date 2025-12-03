import numpy as np

from global_variables import *

# MOTION CONTROL FUNCTIONS ----------------------------------------

def convert_speed_to_tymio_int(v_speed):
  if np.abs(v_speed) > 20e-2:
    print('Demanded speed is outside Thymio range and has been capped')
    return int(np.sign(v_speed) * 500)
  else:
    return int(np.sign(v_speed) * np.floor(np.abs(v_speed) * 500 / (20e-2)))

def heading_correction_command(X, waypoint,N):
  # Returns the corrections to be applied to the commands v_l and v_r corresponding to the control law above for global navigation, computed from the state vector at time i (X), the waypoint coordinates and the horizon N
  # Inputs : X (3x1 numpy array), waypoint (2x1 numpy array), N (int)
  # Outputs : del_v_r (int), del_v_l (int)

  x = X[0]
  y = X[1]
  theta = X[2]

  x_des = waypoint[0]
  y_des = waypoint[1]
  theta_des = np.arctan2(y_des-y, x_des-x)

  print(theta_des - theta)

  del_v_r = 0.5 * (theta_des - theta) * wheel_spacing / (dt * N)
  del_v_l = - del_v_r

  del_v_r = convert_speed_to_tymio_int(del_v_r)
  del_v_l = convert_speed_to_tymio_int(del_v_l)

  return int(del_v_r), int(del_v_l)

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