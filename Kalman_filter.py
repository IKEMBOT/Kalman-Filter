import numpy as np
import matplotlib.pyplot as plt

"""
kalman_filter(): Function that performs one iteration of the Kalman filter.
Inputs:
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

Outputs:
  - x: Updated state estimate (2x1).
       The improved estimate of position and velocity after incorporating the new measurement.
  - P: Updated state covariance estimate (2x2).
       The updated uncertainty in the state estimate.
"""


# Time step and total time
dt = 0.1
total_time = 10
timesteps = int(total_time / dt)

A = np.array([[1, dt],
              [0,  1]])

B = None
u = None
H = np.array([[1, 0]])
Q = np.array([[0.001, 0],
              [0, 0.001]])
R = np.array([[0.1]])
x = np.array([[0],
              [1]])
P = np.array([[1, 0],
              [0, 1]])

true_states = np.zeros((timesteps, 2))
# print(true_states.shape)
measurements = np.zeros((timesteps, 1))

for t in range(timesteps):
    true_states[t] = x.flatten()
    measurements[t] = H @ x + np.random.normal(0, np.sqrt(R))
    x = A @ x

def kalman_filter(A, H, Q, R, z, x, P):
    # Prediction
    x_pred = A @ x
    P_pred = A @ P @ A.T + Q

    # Update
    y = z - H @ x_pred
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    x = x_pred + K @ y
    P = (np.eye(2) - K @ H) @ P_pred

    return x, P

estimated_states = np.zeros((timesteps, 2))

for t in range(timesteps):
    z = measurements[t].reshape(-1, 1)
    x, P = kalman_filter(A, H, Q, R, z, x, P)
    estimated_states[t] = x.flatten()

fig, axs = plt.subplots(1, 2, figsize=(8, 6))
axs[0].plot(true_states[:, 0], label='True position')
axs[0].plot(measurements, label='Noisy measurements', linestyle='--', alpha=0.5)
axs[0].plot(estimated_states[:, 0], label='Estimated position')
axs[0].set_xlabel('Time step')
axs[0].set_ylabel('Position')
axs[0].legend()
axs[0].set_title('Kalman Filter: Position Estimation')

axs[1].plot(true_states[:, 1], label='True velocity')
axs[1].plot(estimated_states[:, 1], label='Estimated velocity')
axs[1].set_xlabel('Time step')
axs[1].set_ylabel('Velocity')
axs[1].legend()
axs[1].set_title('Kalman Filter: Velocity Estimation')

plt.tight_layout()
plt.show()
