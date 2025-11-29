import numpy as np

from global_variables import *

# KALMAN FILTER HELPER FUNCTIONS ----------------------------------------

def convert_thymio_int_to_speed(motor_int):
  return motor_int * 20e-2 / 500

def camera_to_world(frame_size,xy,theta):
  return [xy[0], frame_size[1] - xy[1]], -theta

def motion_model(X,U,dt):
  # Returns the next pose of the Tymio robot predicted by the motion model for an inital pose X = [x[m],y[m],theta[rad]], a control U = [v_r[m/s],v_l[m/s]] and a time step dt
  # Inputs: X (3x1 numpy array), U (2x1 numpy array), dt (positive float)
  # Outputs : X+ (3x1 numpy array)

  x = X[0]
  y = X[1]
  theta = X[2]
  v_r = U[0]
  v_l = U[1]

  x_plus = x + (v_r+v_l)/2 * np.cos(theta) * dt
  y_plus = y + (v_r+v_l)/2 * np.sin(theta) * dt
  theta_plus = theta + (v_r-v_l)/wheel_spacing * dt

  return np.array([x_plus,y_plus,theta_plus]).T

def estimate_next_pose(X_meas,P,U_sens,dt):
  # Returns the next pose of the Tymio robot predicted by the motion model for a measured pose X_meas = [x[m],y[m],theta[rad]], a sensor value for the control input U_sens = [v_r[int],v_l[int]] and a time step dt
  # as well as the uncertainty associated with this estimation
  # Inputs: X_meas (3x1 numpy array), P (3x3 numpy array), U_sens (2x1 numpy array), dt (positive float)
  # Outputs : X_estim (3x1 numpy array), P_estim (3x3 numpy array)

  x = X_meas[0]
  y = X_meas[1]
  theta = X_meas[2]
  v_r = convert_thymio_int_to_speed(U_sens[0])
  v_l = convert_thymio_int_to_speed(U_sens[1])

  X_estim = motion_model(X_meas,np.array([v_r,v_l]).T,dt)

  J = np.array([[1,0,-(v_r+v_l)/2 * np.sin(theta) * dt],[0,1,(v_r+v_l)/2 * np.cos(theta) * dt],[0,0,1]])
  G = np.array([[0.5*np.cos(theta),0.52*np.cos(theta)],[0.5*np.sin(theta),0.5*np.sin(theta)],[dt/wheel_spacing, -dt/wheel_spacing]])

  Q = G @ np.array([[var_v_r,0],[0,var_v_l]]) @ G.T + alpha * np.eye(3)

  P_estim = J @ P @ J.T + Q

  return X_estim, P_estim

def correct_estimation(X_estim,P_estim,Z_meas):
  # Returns the corrected pose of the Tymio robot (based on the EKF) based on the estimated pose (given by the control model) X_estim = [x[m],y[m],theta[rad]], and the measured pose Z_meas = [x_cam[m],y_cam[m],theta_cam[rad]]
  # as well as the uncertainty associated with this corrected position
  # Inputs: X_estim (3x1 numpy array), P_estim (3x3 numpy array), Z_meas (3x1 numpy array)
  # Outputs : X_corr (3x1 numpy array), P_corr (3x3 numpy array)

  Y = Z_meas - X_estim # compute the residual
  Y[2] = (Y[2] + np.pi) % (2*np.pi) - np.pi # wrap the residual angle to ]-pi,pi]

  S = P_estim + np.array([[var_x_cam,0,0],[0,var_y_cam,0],[0,0,var_theta_cam]])
  S_inv = np.linalg.inv(S)

  # compute Mahalanobis distance and abort correction step if the measurement is rejected
  if Y.T @ S_inv @ Y > threshold:
    print('Measurement rejected')
    return X_estim, P_estim

  K = P_estim @ S_inv

  X_corr = X_estim + K @ Y
  P_corr = (np.eye(3) - K) @ P_estim

  return X_corr, P_corr

