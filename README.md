# Path Estimation with Kalman Filter

This project demonstrates how to estimate and smooth the path of a moving object using noisy tracking points (dots). By applying the **Kalman filter**, we can predict and refine the object's trajectory, providing a more accurate path even when the input data is noisy or imprecise.

## Circuit Diagram

Here is the circuit diagram used in the project:

![Circuit Diagram](circuit.png)

## Input Data: Noisy Tracking Points

The input consists of noisy tracking points (dots) representing the object's position over time. These points may appear imprecise or erratic:

![Noisy Tracking Points](input_points.png)

## Path Estimation Result

After applying the Kalman filter to the noisy tracking points, the result is a smooth and accurate estimated path:

![Kalman Filter Path Estimation](estimated_path.png)

## Features:

- **Input**: A series of noisy tracking points (dots).
- **Kalman Filter**: Used to smooth the data and estimate the true path.
- **Output**: A continuous and accurate trajectory.

## How It Works:

1. **Input Data**: A set of dots representing the object's positions at different times.
2. **Kalman Filter**: Predicts the next position based on previous data, then corrects it when new measurements are received.
3. **Path Output**: The Kalman filter generates a smoothed, continuous path that accurately reflects the object's movement.

## Applications:

- Robotics
- Computer Vision
- Autonomous Vehicles
