import cv2
import numpy as np
import yaml
from scipy.interpolate import CubicSpline

# Load configuration from YAML
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# ========== Parameters ==========
common = config["common"]
image_path = common["image_path"]
dt = float(common["dt"])

# KF parameters
Q_kf = np.array(common["Q"]["kf"], dtype=float)
R_kf = np.array(common["R"]["kf"], dtype=float)
P_initial_kf = np.array(common["P_initial"]["kf"], dtype=float)
A_kf = np.array([[1, dt], [0, 1]], dtype=float)
H_kf = np.array(config["kf"]["H"], dtype=float)

# EKF parameters
Q_ekf = np.array(common["Q"]["ekf"], dtype=float)
R_ekf = np.array(common["R"]["ekf"], dtype=float)
P_initial_ekf = np.array(common["P_initial"]["ekf"], dtype=float)

# Visualization settings
thick_dot = int(config["visualization"]["thick_dot"])
thick_kf = int(config["visualization"]["thick_est"])  # Thickness for KF line
thick_ekf = thick_kf + 1  # Thickness for EKF line

# =======================================
# Load the circuit image
circuit_img = cv2.imread(image_path)
circuit_img = cv2.cvtColor(circuit_img, cv2.COLOR_BGR2RGB)
original_img = circuit_img.copy()

# Store clicked points (landmarks)
points = []

# Initial states and covariances
x_kf = np.array([[0.0], [0.0]])  # KF: Initial position and velocity
P_kf = P_initial_kf.copy()

x_ekf = np.array([[0.0], [0.0], [0.0]])  # EKF: Initial x, y, theta
P_ekf = P_initial_ekf.copy()


def click_event(event, x, y, flags, param):
    """Capture mouse click events to record points."""
    global points, circuit_img
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        # Draw a small circle at the clicked point
        cv2.circle(circuit_img, (x, y), thick_dot, (255, 0, 0), -1)  # Blue dot
        cv2.imshow('Circuit', circuit_img)


def kalman_filter(A, H, Q, R, z, x, P):
    """Perform a Kalman Filter update."""
    # Prediction
    x_pred = A @ x
    P_pred = A @ P @ A.T + Q

    # Update
    y = z - H @ x_pred  # Measurement residual
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)  # Kalman gain
    x = x_pred + K @ y  # Corrected state estimate
    P = (np.eye(len(P)) - K @ H) @ P_pred  # Corrected covariance estimate

    return x, P


def state_transition(x, u, dt):
    """Nonlinear state transition for EKF."""
    theta = x[2, 0]
    v, omega = u
    if omega == 0:  # Straight-line motion
        dx = v * dt * np.cos(theta)
        dy = v * dt * np.sin(theta)
        dtheta = 0
    else:  # Rotational motion
        dx = (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
        dy = (v / omega) * (-np.cos(theta + omega * dt) + np.cos(theta))
        dtheta = omega * dt
    return np.array([[x[0, 0] + dx], [x[1, 0] + dy], [x[2, 0] + dtheta]])


def measurement_function(x):
    """Measurement function for EKF."""
    return np.array([[x[0, 0]], [x[1, 0]]])


def jacobian_f(x, u, dt):
    """Compute the Jacobian of the state transition function for EKF."""
    theta = x[2, 0]
    v, omega = u
    if omega == 0:  # Straight-line motion
        Fx = np.array([
            [1, 0, -v * dt * np.sin(theta)],
            [0, 1,  v * dt * np.cos(theta)],
            [0, 0,  1]
        ])
    else:  # Rotational motion
        Fx = np.array([
            [1, 0, (v / omega) * (np.cos(theta + omega * dt) - np.cos(theta))],
            [0, 1, (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))],
            [0, 0, 1]
        ])
    return Fx


def jacobian_h(x):
    """Compute the Jacobian of the measurement function for EKF."""
    return np.array([[1, 0, 0], [0, 1, 0]])


def ekf_update(x, P, z, u, Q, R, dt):
    """Perform an Extended Kalman Filter update step."""
    F = jacobian_f(x, u, dt)
    x_pred = state_transition(x, u, dt)
    P_pred = F @ P @ F.T + Q

    H = jacobian_h(x)
    z_pred = measurement_function(x_pred)
    y = z - z_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x = x_pred + K @ y
    P = (np.eye(len(P)) - K @ H) @ P_pred

    return x, P


def follow_path():
    """Simulate robot following a path using both KF and EKF."""
    global points, circuit_img, x_kf, x_ekf, P_kf, P_ekf

    if len(points) > 1:
        points.append(points[0])  # Close the loop
        x_points = [p[0] for p in points]
        y_points = [p[1] for p in points]
        cs_x = CubicSpline(range(len(x_points)), x_points, bc_type='periodic')
        cs_y = CubicSpline(range(len(y_points)), y_points, bc_type='periodic')
        smooth_path = np.array([(cs_x(i), cs_y(i)) for i in np.linspace(0, len(points) - 1, 500)])

        x_kf = np.array([[smooth_path[0, 0]], [0.0]])
        P_kf = P_initial_kf.copy()

        x_ekf = np.array([[smooth_path[0, 0]], [smooth_path[0, 1]], [0.0]])
        P_ekf = P_initial_ekf.copy()

        u = [1.0, 0.1]  # Velocity and angular velocity

        est_kf = np.zeros_like(smooth_path)
        est_ekf = np.zeros_like(smooth_path)

        for t, z in enumerate(smooth_path):
            z_kf = np.array([[z[0]]])
            x_kf, P_kf = kalman_filter(A_kf, H_kf, Q_kf, R_kf, z_kf, x_kf, P_kf)
            est_kf[t, 0], est_kf[t, 1] = x_kf[0, 0], z[1]

            z_ekf = np.array([[z[0]], [z[1]]])
            x_ekf, P_ekf = ekf_update(x_ekf, P_ekf, z_ekf, u, Q_ekf, R_ekf, dt)
            est_ekf[t, 0], est_ekf[t, 1] = x_ekf[0, 0], x_ekf[1, 0]

        for i in range(1, len(smooth_path)):
            pt1_kf = tuple(map(int, est_kf[i - 1]))
            pt2_kf = tuple(map(int, est_kf[i]))
            cv2.line(circuit_img, pt1_kf, pt2_kf, (0, 255, 0), thick_kf)

            pt1_ekf = tuple(map(int, est_ekf[i - 1]))
            pt2_ekf = tuple(map(int, est_ekf[i]))
            cv2.line(circuit_img, pt1_ekf, pt2_ekf, (0, 0, 255), thick_ekf)

        cv2.putText(circuit_img, "Blue Dot: Waypoints", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(circuit_img, "Green Line: KF Path", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(circuit_img, "Blue Line: EKF Path", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('Circuit with Path Following', circuit_img)
    else:
        print("At least two points are required to create a path.")


cv2.imshow('Circuit', circuit_img)
cv2.setMouseCallback('Circuit', click_event)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 13:
        follow_path()
    elif key == 27:
        break

cv2.destroyAllWindows()
