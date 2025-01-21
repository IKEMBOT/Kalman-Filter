import cv2
import numpy as np
import yaml
from scipy.interpolate import CubicSpline

"""
Extended Kalman Filter (EKF) Path Following
===========================================
This script implements an EKF-based path-following simulation where the robot follows waypoints 
clicked by the user. A cubic spline is used to interpolate a smooth path, and the EKF updates 
its state and covariance at each step.

Variable Descriptions:
----------------------
- dt: Time step (float) - interval between state updates.
- image_path: Path to the circuit image (string).
- thick_dot: Thickness of the red dots marking waypoints (int).
- thick_est: Thickness of the green line representing the EKF path (int).
- Q: Process noise covariance matrix (3x3, float).
- R: Measurement noise covariance matrix (2x2, float).
- P_initial: Initial state covariance matrix (3x3, float).
- H: Measurement matrix (2x3, float) - maps the state to measurements.
- x: State vector (3x1, float) - [x-position, y-position, angle].
- P: State covariance matrix (3x3, float) - uncertainty in the state.
- u: Control inputs (list) - [velocity, angular velocity].
"""


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

common = config["common"]
dt = float(common["dt"])
image_path = common["image_path"]

thick_dot = int(config["visualization"]["thick_dot"])
thick_est = int(config["visualization"]["thick_est"])

Q = np.array(common["Q"]["ekf"], dtype=float)
R = np.array(common["R"]["ekf"], dtype=float)
P_initial = np.array(common["P_initial"]["ekf"], dtype=float)
H = np.array(config["ekf"]["H"], dtype=float)

circuit_img = cv2.imread(image_path)
circuit_img = cv2.cvtColor(circuit_img, cv2.COLOR_BGR2RGB)
original_img = circuit_img.copy()

points = []

x = np.array([[0.0], [0.0], [0.0]])
P = P_initial.copy()

def click_event(event, x, y, flags, param):
    global points, circuit_img
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(circuit_img, (x, y), thick_dot, (255, 0, 0), -1)
        cv2.imshow('Circuit', circuit_img)

def state_transition(x, u, dt):
    theta = x[2, 0]
    v, omega = u
    if omega == 0:
        new_x = x[0, 0] + v * dt * np.cos(theta)
        new_y = x[1, 0] + v * dt * np.sin(theta)
        new_theta = theta
    else:
        new_x = x[0, 0] + (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
        new_y = x[1, 0] + (v / omega) * (-np.cos(theta + omega * dt) + np.cos(theta))
        new_theta = theta + omega * dt
    return np.array([[new_x], [new_y], [new_theta]])

def measurement_function(x):
    return np.array([[x[0, 0]], [x[1, 0]]])

def jacobian_f(x, u, dt):
    theta = x[2, 0]
    v, omega = u
    if omega == 0:
        Fx = np.array([
            [1, 0, -v * dt * np.sin(theta)],
            [0, 1,  v * dt * np.cos(theta)],
            [0, 0,  1]
        ])
    else:
        Fx = np.array([
            [1, 0, (v / omega) * (np.cos(theta + omega * dt) - np.cos(theta))],
            [0, 1, (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))],
            [0, 0, 1]
        ])
    return Fx

def jacobian_h(x):
    return H

def ekf_update(x, P, z, u, Q, R, dt):
    F = jacobian_f(x, u, dt)
    x_pred = state_transition(x, u, dt)
    P_pred = F @ P @ F.T + Q
    z_pred = measurement_function(x_pred)
    y = z - z_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x = x_pred + K @ y
    P = (np.eye(len(P)) - K @ H) @ P_pred
    return x, P

def follow_path():
    global points, circuit_img, x, P
    if len(points) > 1:
        points.append(points[0])
        x_points = [p[0] for p in points]
        y_points = [p[1] for p in points]
        cs_x = CubicSpline(range(len(x_points)), x_points, bc_type='periodic')
        cs_y = CubicSpline(range(len(y_points)), y_points, bc_type='periodic')
        smooth_path = np.array([(cs_x(i), cs_y(i)) for i in np.linspace(0, len(points) - 1, 500)])
        x = np.array([[smooth_path[0, 0]], [smooth_path[0, 1]], [0.0]])
        P = P_initial.copy()
        v, omega = 1.0, 0.1
        u = [v, omega]
        estimated_states = np.zeros_like(smooth_path)
        for t, z in enumerate(smooth_path):
            z = np.array([[z[0]], [z[1]]])
            x, P = ekf_update(x, P, z, u, Q, R, dt)
            estimated_states[t] = [x[0, 0], x[1, 0]]
        for i in range(1, len(estimated_states)):
            pt1_est = (int(estimated_states[i - 1][0]), int(estimated_states[i - 1][1]))
            pt2_est = (int(estimated_states[i][0]), int(estimated_states[i][1]))
            cv2.line(circuit_img, pt1_est, pt2_est, (0, 255, 0), thick_est)
        cv2.putText(circuit_img, "Red Dot: Waypoints", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(circuit_img, "Green Line: EKF Path", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Circuit with EKF Path Following', circuit_img)

cv2.imshow('Circuit', circuit_img)
cv2.setMouseCallback('Circuit', click_event)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 13:
        follow_path()
    elif key == 27:
        break

cv2.destroyAllWindows()
