import cv2
import numpy as np
import yaml
from scipy.interpolate import CubicSpline

"""
  - A: State transition matrix (2x2).
       Defines how the state (position and velocity) evolves over time.
  - H: Measurement matrix (1x2).
       Maps the true state to the measured state (e.g., extracts position from [position, velocity]).
  - Q: Process noise covariance (2x2).
       Represents uncertainty in the system's dynamics (affects both position and velocity).
  - R: Measurement noise covariance (1x1).
       Represents uncertainty in the position measurement.
  - z: Current measurement (1x1).
       The observed noisy position at the current time step.
  - x: Current state estimate (2x1).
       The best estimate of position and velocity before incorporating the new measurement.
  - P: Current state covariance estimate (2x2).
       Represents uncertainty in the current state estimate.
  - x: Updated state estimate (2x1).
       The improved estimate of position and velocity after incorporating the new measurement.
  - P: Updated state covariance estimate (2x2).
       The updated uncertainty in the state estimate.
"""

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


common = config["common"]
image_path = common["image_path"]
dt = float(common["dt"])  # Time step
Q = np.array(common["Q"]["kf"], dtype=float)  # KF process noise covariance (2x2)
R = np.array(common["R"]["kf"], dtype=float)  # KF measurement noise covariance (1x1)
P_initial = np.array(common["P_initial"]["kf"], dtype=float)  # KF initial state covariance (2x2)
A = np.array(config["kf"]["A"], dtype=float)  # State transition matrix (2x2)
H = np.array(config["kf"]["H"], dtype=float)  # Measurement matrix (1x2)

thick_dot = int(config["visualization"]["thick_dot"])  # Thickness for dots
thick_est = int(config["visualization"]["thick_est"])  # Thickness for the estimated path

circuit_img = cv2.imread(image_path)
circuit_img = cv2.cvtColor(circuit_img, cv2.COLOR_BGR2RGB)
original_img = circuit_img.copy()  # Keep a copy of the original image

points = []
x = np.array([[0.0],  
              [0.0]]) 
P = P_initial.copy()

def click_event(event, x, y, flags, param):
    """Capture mouse click events to record points."""
    global points, circuit_img
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        # Draw a small circle at the clicked point
        cv2.circle(circuit_img, (x, y), thick_dot, (0, 0, 255), -1)  # Red dot
        cv2.imshow('Circuit', circuit_img)

def kalman_filter(A, H, Q, R, z, x, P):
    """Perform a Kalman Filter update."""
    x_pred = A @ x
    P_pred = A @ P @ A.T + Q

    # Update
    y = z - H @ x_pred  
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)  
    x = x_pred + K @ y  
    P = (np.eye(len(P)) - K @ H) @ P_pred  

    return x, P

def follow_path():
    """Simulate robot following a path using Kalman Filter."""
    global points, circuit_img, x, P

    if len(points) > 1:
        points.append(points[0])
        x_points = [p[0] for p in points]
        y_points = [p[1] for p in points]
        cs_x = CubicSpline(range(len(x_points)), x_points, bc_type='periodic')  # Ensure periodicity
        cs_y = CubicSpline(range(len(y_points)), y_points, bc_type='periodic')  # Ensure periodicity
        smooth_path = np.array([(cs_x(i), cs_y(i)) for i in np.linspace(0, len(points) - 1, 500)])

        x = np.array([[smooth_path[0, 0]], 
                      [0.0]])  
        P = P_initial.copy()  

        estimated_states = np.zeros_like(smooth_path)
        for t, z in enumerate(smooth_path):  
            z = np.array([[z[0]]], dtype=float)
            x, P = kalman_filter(A, H, Q, R, z, x, P)
            estimated_states[t, 0] = x[0, 0]
            estimated_states[t, 1] = smooth_path[t, 1] 

        for i in range(1, len(estimated_states)):
            pt1_est = (int(estimated_states[i - 1][0]), int(estimated_states[i - 1][1]))
            pt2_est = (int(estimated_states[i][0]), int(estimated_states[i][1]))
            cv2.line(circuit_img, pt1_est, pt2_est, (0, 255, 0), thick_est) 


        cv2.putText(circuit_img, "Red Dot: Waypoints", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(circuit_img, "Green Line: Estimated Path", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Circuit with Path Following', circuit_img)
    else:
        print("At least two points are required to create a path.")

cv2.imshow('Circuit', circuit_img)
cv2.setMouseCallback('Circuit', click_event)

print("Instructions:")
print("1. Click points on the image to select waypoints.")
print("2. Press 'Enter' to show the result on the same frame.")
print("3. Press 'Esc' to exit.")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 13:  
        follow_path()
    elif key == 27:  
        break

cv2.destroyAllWindows()
